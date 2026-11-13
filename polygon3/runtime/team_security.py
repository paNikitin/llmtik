"""Ограничения процесса роли в Linux, только для C8. Модуль ничего не импортирует
из app и из предметной области.

Landlock с ABI не ниже 4 защищает файлы. Seccomp независимо от него запрещает
межпроцессное взаимодействие, сеть и создание процессов, в том числе на ABI 4 и 5.
Вызывать только в новом
однопоточном процессе роли до импорта кода студента.
"""
from __future__ import annotations

import ctypes
import errno
import os
import struct
import sys
import sysconfig
from pathlib import Path


class SecurityUnavailable(RuntimeError):
    pass


PR_SET_DUMPABLE = 4
PR_GET_DUMPABLE = 3
PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
MIN_LANDLOCK_ABI = 4


def _libc():
    lib = ctypes.CDLL(None, use_errno=True)
    lib.prctl.restype = ctypes.c_int
    lib.syscall.restype = ctypes.c_long
    return lib


def require_nondumpable() -> None:
    if not sys.platform.startswith("linux"):
        raise SecurityUnavailable("required isolation needs Linux")
    lib = _libc()
    if lib.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise SecurityUnavailable("PR_SET_DUMPABLE failed")
    if lib.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise SecurityUnavailable("process remains dumpable")


def require_unprivileged() -> None:
    if not sys.platform.startswith("linux") or os.getuid() == 0 or os.geteuid() == 0:
        raise SecurityUnavailable("required isolation needs a non-root Linux uid")
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        for name in ("CapEff", "CapPrm", "CapInh", "CapAmb"):
            if int(fields[name].strip(), 16) != 0:
                raise SecurityUnavailable("required isolation needs empty process capabilities")
    except (OSError, KeyError, ValueError) as exc:
        raise SecurityUnavailable("cannot verify process capabilities") from exc


def close_extra_fds() -> None:
    """После доверенного запуска открытыми остаются только дескрипторы 0, 1 и 2,
    и все три должны быть анонимными каналами. Унаследованные каналы
    межпроцессного взаимодействия закрываются."""
    try:
        fds = [int(p) for p in os.listdir("/proc/self/fd")]
    except OSError as exc:
        raise SecurityUnavailable("cannot enumerate inherited descriptors") from exc
    for fd in fds:
        if fd > 2:
            try:
                os.close(fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    import stat
    for fd in (0, 1, 2):
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise SecurityUnavailable("role stdio must be anonymous pipes")


def _call(lib, number, *args):
    result = lib.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _landlock(workdir: str, read_only: list[str]) -> int:
    # Эти номера системных вызовов совпадают в поддерживаемых Linux на x86_64 и aarch64.
    import platform
    if platform.machine().lower() not in ("x86_64", "amd64", "aarch64", "arm64"):
        raise SecurityUnavailable("unsupported Landlock syscall architecture")
    lib = _libc()
    try:
        abi = _call(lib, 444, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
        if abi < MIN_LANDLOCK_ABI:
            raise SecurityUnavailable("C8 requires Landlock ABI >= 4")
        fs = (1 << (16 if abi >= 5 else 15)) - 1
        # Области ограничений из ABI 6 добавляют ещё один слой защиты.
        # На ABI 4 и 5 их отсутствие не мешает запуску.
        fields = [fs, 3] + ([3] if abi >= 6 else [])
        attr = struct.pack("=" + "Q" * len(fields), *fields)
        rulefd = _call(lib, 444, ctypes.create_string_buffer(attr), ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
        try:
            def allow(path, rights):
                try:
                    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                except FileNotFoundError:
                    return
                try:
                    import stat
                    if not stat.S_ISDIR(os.fstat(fd).st_mode):
                        rights &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)
                    packed = struct.pack("=Qi", rights & fs, fd)
                    _call(lib, 445, ctypes.c_int(rulefd), ctypes.c_int(1),
                          ctypes.create_string_buffer(packed), ctypes.c_uint32(0))
                finally:
                    os.close(fd)
            # Право на исполнение не выдаётся нигде, даже для файлов интерпретатора,
            # открытых только на чтение.
            for path in read_only:
                allow(path, (1 << 2) | (1 << 3))
            rw = ((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
                  | (1 << 7) | (1 << 8) | (1 << 12) | (1 << 13) | (1 << 14))
            allow(workdir, rw)  # нельзя создавать устройства, сокеты и FIFO, нельзя исполнять файлы
            allow("/dev/null", (1 << 1) | (1 << 2))
            for path in ("/dev/urandom", "/dev/random", "/dev/zero"):
                allow(path, 1 << 2)
            if lib.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
                raise SecurityUnavailable("cannot set no_new_privs")
            if lib.prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1:
                raise SecurityUnavailable("cannot verify no_new_privs")
            # Правил для сетевых портов нет, поэтому connect и bind запрещены на всех портах.
            _call(lib, 446, ctypes.c_int(rulefd), ctypes.c_uint32(0))
        finally:
            os.close(rulefd)
        return abi
    except OSError as exc:
        raise SecurityUnavailable("Landlock setup failed") from exc


# Запрет задаётся по имени системного вызова. libseccomp переводит имя в номер
# для архитектуры текущей машины. Старых имён и имён, которые есть только на
# отдельных архитектурах, может не оказаться, их пропускаем. Вызовы из
# ESSENTIAL_SYSCALLS должны быть найдены, иначе запуск прерывается
# до загрузки кода студента.
DENIED_SYSCALLS = tuple("""
 socket socketpair socketcall connect bind listen accept accept4 shutdown
 sendto recvfrom sendmsg recvmsg sendmmsg recvmmsg getsockname getpeername
 setsockopt getsockopt
 ipc shmget shmat shmdt shmctl semget semop semtimedop semctl msgget msgsnd
 msgrcv msgctl mq_open mq_unlink mq_timedsend mq_timedreceive mq_notify
 mq_getsetattr
 ptrace process_vm_readv process_vm_writev pidfd_open pidfd_getfd
 kill tkill tgkill pidfd_send_signal rt_sigqueueinfo rt_tgsigqueueinfo
 fork vfork clone clone3 execve execveat unshare setns setsid setpgid
 prctl seccomp io_uring_setup io_uring_enter io_uring_register
 perf_event_open bpf keyctl add_key request_key
 mount umount umount2 pivot_root chroot move_mount open_tree fsopen fsconfig
 fsmount fspick mount_setattr open_by_handle_at name_to_handle_at
 fanotify_init fanotify_mark userfaultfd memfd_create memfd_secret
 mknod mknodat acct reboot kexec_load kexec_file_load init_module finit_module
 delete_module swapon swapoff quotactl iopl ioperm modify_ldt personality
 setuid setgid setreuid setregid setresuid setresgid setfsuid setfsgid
 chmod fchmod fchmodat fchmodat2 chown fchown lchown fchownat
 chown32 fchown32 lchown32 utime utimes futimesat utimensat utimensat_time64
 setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr
 capset setgroups ioctl vhangup
 setrlimit prlimit64 setpriority sched_setparam sched_setscheduler
 sched_setaffinity sched_setattr ioprio_set kcmp migrate_pages move_pages
 process_madvise process_mrelease
 flock fcntl fcntl64 inotify_init inotify_init1 inotify_add_watch
 inotify_rm_watch futex futex_time64 futex_waitv futex_wake futex_wait
 futex_requeue get_robust_list set_robust_list
""".split())
ESSENTIAL_SYSCALLS = frozenset("""
 socket socketpair connect ptrace process_vm_readv process_vm_writev
 pidfd_getfd kill tgkill pidfd_send_signal clone clone3 execve unshare setns
 setsid setpgid prctl seccomp execveat pidfd_open ioctl io_uring_setup
 perf_event_open bpf keyctl shmget semget msgget fcntl futex
 fchmod fchmodat fchmodat2 fchown fchownat utimensat setxattr fsetxattr
 prlimit64 setpriority sched_setparam sched_setscheduler sched_setaffinity
""".split())


def _load_seccomp():
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        lib.seccomp_arch_native.argtypes = []
        lib.seccomp_arch_native.restype = ctypes.c_uint32
        lib.seccomp_syscall_resolve_name_arch.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name_arch.restype = ctypes.c_int
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
        lib.seccomp_rule_add.restype = ctypes.c_int
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_load.restype = ctypes.c_int
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_release.restype = None
        return lib
    except (OSError, AttributeError) as exc:
        raise SecurityUnavailable("libseccomp2 is required for C8") from exc


def _seccomp(lib) -> int:
    arch = lib.seccomp_arch_native()
    if not arch:
        raise SecurityUnavailable("cannot resolve native seccomp architecture")
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW; вызов с чужой архитектурой убивает процесс
    if not ctx:
        raise SecurityUnavailable("seccomp_init failed")
    count = 0
    try:
        for name in DENIED_SYSCALLS:
            number = lib.seccomp_syscall_resolve_name_arch(arch, name.encode("ascii"))
            if number < 0:
                if name in ESSENTIAL_SYSCALLS:
                    raise SecurityUnavailable("essential syscall unresolved: " + name)
                continue
            result = lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0)
            if result != 0:
                raise SecurityUnavailable("seccomp rule failed: " + name)
            count += 1
        if lib.seccomp_load(ctx) != 0:
            raise SecurityUnavailable("seccomp_load failed")
    finally:
        lib.seccomp_release(ctx)
    return count


def confine_role(workdir: str) -> dict:
    """Ставит необратимые ограничения C8 до загрузки присланного файла."""
    require_unprivileged()
    require_nondumpable()
    lib = _load_seccomp()  # динамический загрузчик должен отработать до ограничения доступа к файлам
    close_extra_fds()
    work = Path(workdir).resolve(strict=True)
    paths = {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")}
    # Разделяемые библиотеки интерпретатора. Не /etc, не весь /usr, не /proc и не /sys.
    paths.update(("/lib", "/lib64", "/usr/lib", "/usr/local/lib"))
    # В /usr/local/lib часто лежат стандартная библиотека и libpython. Читать оттуда
    # безопасно, только если личный рабочий каталог вызывающего лежит вне этих
    # общих каталогов с кодом.
    read_only = sorted(str(Path(p).resolve()) for p in paths if p and Path(p).exists())
    for path in read_only:
        if work == Path(path) or Path(path) in work.parents:
            raise SecurityUnavailable("role workdir overlaps shared read-only libraries")
    abi = _landlock(str(work), read_only)
    # В дочернем процессе не должно остаться пути импорта к коду сервера и предметной области.
    sys.path[:] = [str(work)] + [p for p in sys.path if p and any(
        Path(p).resolve() == Path(ro) or Path(ro) in Path(p).resolve().parents for ro in read_only)]
    rules = _seccomp(lib)
    return {"landlock_abi": abi, "seccomp": True, "seccomp_rules": rules,
            "nondumpable": True, "no_new_privs": True, "connect_ports": []}
