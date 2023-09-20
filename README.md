# 评分卡模型转PMML部署

![图片](https://itlubber.art/upload/2023/02/%E5%9B%BE%E7%89%87.png)

## 简介

`scorecard2pmml` 提供 `card2pmml` 方法支持用户将 `toad`, `scorecardpy` 以及 离线的评分卡文件 转换为 `PMML` 文件进行评分卡模型的部署

## 交流

<table style="text-align:center !important;border=0;">
    <tr>
        <td>
            <span>微信: itlubber</span>
        </td>
        <td>
            <span>微信公众号: itlubber_art</span>
        </td>
    </tr>
    <tr>
        <td>
            <img src="https://itlubber.art//upload/itlubber.png" alt="itlubber.png" width="50%" border=0/>
        </td>
        <td>
            <img src="https://itlubber.art//upload/itlubber_art.png" alt="itlubber_art.png" width="50%" border=0/>
        </td>
    </tr>
</table>

## 项目结果

```base
itlubber@itlubber:~/workspace/scorecard2pmml$ tree .
.
├── LICENSE                     # 项目开源许可证书
├── README.md                   # 说明文档
├── requirements.txt            # 项目依赖包
├── scorecard2pmml.py           # 评分卡模型转PMML文件功能实现
└── test.py                     # 测试脚本

0 directories, 5 files
```


## 食用方法

0. 样例数据

```python
target = "creditability"
data = sc.germancredit()
data[target] = data[target].map({"good": 0, "bad": 1})
```

1. 生成评分卡变量分箱表

```python
# 使用 toad 生成的评分卡变量分箱信息
data_selected = toad.selection.select(data, target=target, empty=0.5, iv=0.05, corr=0.7)
c = toad.transform.Combiner()
c.fit(data_selected, y=target, method='chi', min_samples=0.05)
transer = toad.transform.WOETransformer()
data_woe = transer.fit_transform(c.transform(data_selected), data_selected[target], exclude=target)
final_data = toad.selection.stepwise(data_woe,target=target, estimator='ols', direction='both', criterion='aic')
card = toad.ScoreCard(combiner=c, transer=transer)
card.fit(final_data.drop(columns=[target]), final_data[target])

# scorecard_feature_bins = card.export(to_frame=True)
scorecard_feature_bins = card2bins(card)

print(scorecard_feature_bins)


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


# 读取excel文件转PMML: 必须包含三列:["变量名", "分箱", "对应分数"], 保证顺序一致
scorecard_feature_bins.to_excel("scorecard.xlsx", sheet_name="card", index=False)

scorecard_feature_bins = pd.read_excel("scorecard.xlsx", sheet_name="card")
print(scorecard_feature_bins)
```

2. 转换 `PMML` 文件

```python
card2pmml(scorecard_feature_bins, pmml="scorecard.pmml")
```

3. 结果验证

```python
from pypmml import Model


model = Model.fromFile("scorecard.pmml")

data["score"] = model.predict(data[model.inputNames]).values
```
