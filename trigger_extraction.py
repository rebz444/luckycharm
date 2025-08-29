import pandas as pd
import os

# File path
file_path = "/Users/rebekahzhang/data/behavior_data/exp2/2025-08-25_11-21-36_RZ074/events_2025-08-25_11-21-36_RZ074.txt"

# Desktop path
desktop_path = os.path.expanduser("~/Desktop")
df = pd.read_csv(file_path)
trigger_rows = df[df['key'] == 'trigger']

base_name = "triggers_2025-08-25_11-21-36_RZ074"
output_file = os.path.join(desktop_path, f"{base_name}.csv")
trigger_rows.to_csv(output_file, index=False)
