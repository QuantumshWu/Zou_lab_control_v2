"""Where a pulse server listens, and where a client looks for one.

Said once, and said HERE rather than inside the server, for two reasons.

The number itself was said five times in four packages -- the server default,
the client default, the CLI default, a widget's placeholder and an apparatus
form default -- so changing the port meant finding them all.

And the module that runs the server is not the place to keep it.  Putting it
there meant the package had to import the server to publish the constant, so
``python -m zlc_pulse.remote`` loaded that module twice: once as a side effect
of importing the package, once as __main__.  Python says exactly that, on the
server launcher, every time it starts:

    RuntimeWarning: 'zlc_pulse.remote' found in sys.modules after import of
    package 'zlc_pulse', but prior to execution of 'zlc_pulse.remote'

A constant should not oblige anyone to load a socket server to read it.
"""

from __future__ import annotations


__all__ = ["DEFAULT_BIND_HOST", "DEFAULT_HOST", "DEFAULT_PORT"]

#: The port a pulse server listens on and a client dials.
DEFAULT_PORT = 18861

#: What a server binds by default: every interface, because the board is often
#: on a different machine from the operator.
DEFAULT_BIND_HOST = "0.0.0.0"

#: Where a client looks by default: this machine, because it usually is.
DEFAULT_HOST = "127.0.0.1"
