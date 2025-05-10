from fusebuild import *

shell_action(
    name="someaction",
    cmd="echo something > $OUTPUT_DIR/output1.txt",
    category="dontbuilddirectly",
    tmp=None,
)
