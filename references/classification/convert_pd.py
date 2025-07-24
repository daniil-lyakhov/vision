import pandas as pd

with open("result.txt", "r") as f:
    lines = f.readlines()

breakpoint()
data = []
for line in lines:
    model, w, nncf_used, bc, result, *_ = line.split(",")
    if result != " -1":
        result = result.split()[2]
    data.append(list(map(lambda x: x.strip(), [model, w, nncf_used, bc, result])))
df = pd.DataFrame(data, columns=["Model", "Weights", "NNCF", "BC", "Acc@1"])

def result_category(row):
    if row['NNCF'] == "False":
        return 'TorchAO'
    elif row['BC'] == "True":
        return 'NNCF+BC'
    return 'NNCF'

df['Category'] = df.apply(result_category, axis=1)

breakpoint()
# Pivot the DataFrame
pivot_df = df.pivot(index=['Model', "Weights"], columns='Category', values='Acc@1').reset_index()

# Optional: Sort columns for consistent order
#pivot_df = pivot_df[['Model', 'Weights', 'TorchAO', 'NNCF', 'NNCF+BC']]
pivot_df = pivot_df[['Model', 'Weights', 'NNCF', 'NNCF+BC']]
print(pivot_df)
pivot_df.to_csv("pivot_result.csv")
#print(lines)
