# Ina kernel lab

This is a rebuildable experimental workspace, not a production operating-system
tree. The initial baseline is the current 6.18 long-term Linux release from
kernel.org. A source archive is not admitted until its digest and kernel.org
signature have both been verified.

VM experiments are offline by default and use QEMU TCG without host files,
shared folders, USB, KVM, writable base disks, or production boot authority.
Each run gets one bounded attempt. Networked package acquisition, distribution
comparison images, hardware acceleration, and persistent writable disks require
separate reviewed capability grants.

Comparison with other distributions keeps functional, security, resource,
recovery, reproducibility and update-latency evidence separate. There is no
single “most secure OS” score and no automatic promotion or host installation.
