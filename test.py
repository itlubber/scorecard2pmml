import warnings
import numpy as np
import pandas as pd
import toad
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression

from scorecard2pmml import card2bins, card2pmml


warnings.filterwarnings("ignore")


target = "creditability"
data = sc.germancredit()
data[target] = data[target].map({"good": 0, "bad": 1})


# 使用 toad 生成的评分卡模型转PMML文件
data_selected = toad.selection.select(data, target=target, empty=0.5, iv=0.05, corr=0.7)
c = toad.transform.Combiner()
c.fit(data_selected, y=target, method='chi', min_samples=0.05)
transer = toad.transform.WOETransformer()
data_woe = transer.fit_transform(c.transform(data_selected), data_selected[target], exclude=target)
final_data = toad.selection.stepwise(data_woe,target=target, estimator='ols', direction='both', criterion='aic')
card = toad.ScoreCard(combiner=c, transer=transer)
card.fit(final_data.drop(columns=[target]), final_data[target])

# toad_feature_bins = card.export(to_frame=True)

scorecard_feature_bins = card2bins(card)
print(scorecard_feature_bins)

card2pmml(scorecard_feature_bins, pmml="toad_scorecard.pmml")


# 使用 scorecardpy 生成的评分卡模型转PMML文件
data_selected = sc.var_filter(data, y=target)
bins = sc.woebin(data_selected, y=target)
data_woe = sc.woebin_ply(data_selected, bins)
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(data_woe.drop(columns=target), data_woe[target])
card = sc.scorecard(bins, lr, list(data_woe.drop(columns=target).columns))

# scorecardpy_feature_bins = pd.concat({k: v for k, v in card.items() if k != "basepoints"}.values())

scorecard_feature_bins = card2bins(card)
print(scorecard_feature_bins)

card2pmml(scorecard_feature_bins, pmml="scorecardpy_scorecard.pmml")


# 读取excel文件转PMML
scorecard_feature_bins.to_excel("scorecard.xlsx", sheet_name="card", index=False)


scorecard_feature_bins = pd.read_excel("scorecard.xlsx", sheet_name="card")
card2pmml(scorecard_feature_bins, pmml="offline_scorecard.pmml")
