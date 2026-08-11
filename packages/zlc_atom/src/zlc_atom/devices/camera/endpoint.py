"""Where a camera server listens, and where a client looks for one.

Said once, and said HERE rather than inside the server, for the same two
reasons ``zlc_pulse.endpoint`` exists.

The number would otherwise be said four times -- the server's bind default,
the client's dial default, the CLI default and the ``camera.remote`` authoring
schema default -- so changing the port would mean finding them all.

And the module that runs the server is not the place to keep it.  Reading a
constant must not oblige anyone to load a socket server and a numpy adapter
stack; and if the package ever published the constant by importing the server
module, ``python -m zlc_atom.devices.camera.remote`` would load that module
twice and Python would say so on every server start.
"""

from __future__ import annotations


__all__ = ["DEFAULT_BIND_HOST", "DEFAULT_HOST", "DEFAULT_PORT"]

#: The port a camera server listens on and a client dials.  One above the
#: pulse server's 18861, so both servers can share a lab machine.
DEFAULT_PORT = 18862

#: What a server binds by default: every interface, because the camera is on
#: a different machine from the operator -- that is the point of the server.
DEFAULT_BIND_HOST = "0.0.0.0"

#: Where a client looks by default: this machine, for loopback and tests.
DEFAULT_HOST = "127.0.0.1"
