rm submission.tar.gz

find . -type f | grep pyc$ | xargs rm
find . | grep swp$  | xargs rm
find . | grep un~$  | xargs rm

tar -czf submission.tar.gz *

open .
