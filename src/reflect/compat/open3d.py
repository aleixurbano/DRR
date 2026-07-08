import contextlib
import io
import os
import sys


def _stream_fileno(stream):
    if stream is None:
        return None
    try:
        return stream.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError):
        return None


@contextlib.contextmanager
def _quiet_import():
    """Silence Open3D's import-time Jupyter/WebRTC startup chatter."""
    sink = open(os.devnull, "w")
    stdout_fd = _stream_fileno(getattr(sys, "__stdout__", None)) or _stream_fileno(sys.stdout)
    stderr_fd = _stream_fileno(getattr(sys, "__stderr__", None)) or _stream_fileno(sys.stderr)
    saved_fds = []
    try:
        if stdout_fd is not None:
            saved_fds.append((stdout_fd, os.dup(stdout_fd)))
            os.dup2(sink.fileno(), stdout_fd)
        if stderr_fd is not None:
            saved_fds.append((stderr_fd, os.dup(stderr_fd)))
            os.dup2(sink.fileno(), stderr_fd)

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        for target_fd, saved_fd in reversed(saved_fds):
            try:
                os.dup2(saved_fd, target_fd)
            finally:
                os.close(saved_fd)
        sink.close()


with _quiet_import():
    import open3d as o3d


__all__ = ["o3d"]
