import pandas as pd



sub1 = pd.read_csv("../sub/gbm_ltislit.csv")
sub2 = pd.read_csv("../sub/gbm_ltislit1.csv")

print(sub1)
print(sub2)

sub_ens = (sub1 + sub2) / 2
print(sub_ens)

