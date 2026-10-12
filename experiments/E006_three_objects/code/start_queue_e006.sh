#!/bin/bash
exec >> ~/planck/logs/e006_queue.out 2>&1
# E006: the wrapper pcdetach.sh starts on the PC (notes.txt PC QUEUE). It runs queue_e006.sh from the synced working
# copy and touches a DONE marker when the queue script returns (whatever its exit code).
echo "start_queue_e006 $(date '+%F %T')"
bash ~/planck/dev/e006/experiments/E006_three_objects/code/queue_e006.sh
rc=$?
echo "queue_e006 exit $rc $(date '+%F %T')"
touch ~/planck/logs/e006_queue.DONE
exit $rc
