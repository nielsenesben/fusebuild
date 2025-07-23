from fusebuild import shell_action

shell_action(name="A", cmd="cp $OUTPUT_DIR/../C/file.txt file.txt", category="willfail")

shell_action(name="B", cmd="cp $OUTPUT_DIR/../A/file.txt file.txt", category="willfail")

shell_action(name="C", cmd="cp $OUTPUT_DIR/../B/file.txt file.txt", category="willfail")
