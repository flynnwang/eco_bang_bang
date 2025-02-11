rm submission.tar.gz

find . -type f | grep pyc | xargs rm

tar -czf submission.tar.gz *

open .
