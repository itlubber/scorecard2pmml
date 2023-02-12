import warnings
import re
import toad
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer


warnings.filterwarnings("ignore")


def card2bins(card):
    if isinstance(card, toad.ScoreCard):
        return card.export(to_frame=True)
    elif isinstance(card, dict):
        return pd.concat({k: v for k, v in card.items() if k != "basepoints"}.values())

    raise Exception("card only support toad or scorecardpy")


def extract_elements(bin_var):
    pattern = re.compile('^\[(-?\d+|\-inf)(,|\s~\s)(-?\d+|inf)\)$')
    match = pattern.match(bin_var)

    if match:
        return match.group(1), match.group(3)
    else:
        return None, None


def card2pmml(scorecard_feature_bins: pd.DataFrame, pmml: str="scorecard.pmml", n_samples: int=20, debug: bool=False):
    """
    将评分卡模型转换为PMML文件

    Args:
        scorecard_feature_bins: 变量分箱信息表, 包含三列: ["变量名", "分箱", "对应分数"], 保证顺序一致即可
        pmml: 评分卡模型转PMML文件后保存的路径, 默认存储至当前路径下的 scorecard.pmml 文件, 未判断文件保存路径是否存在, 故如果保存至某个目录需要保证该目录存在
        n_samples: 可忽略的参数, 用于生成模拟数据训练 pipeline
        debug: 是否开启调试模式, 默认False, 为True时输出更多信息
    """
    feature_bin_vars = scorecard_feature_bins.copy()
    headers = feature_bin_vars.columns.tolist()

    assert len(headers) == 3

    mapper = []
    # samples = {}
    # n_samples = 20

    for var, bin_vars in feature_bin_vars.groupby(headers[0]):
        if sum([bin_var.startswith("[") and bin_var.endswith(")") for bin_var in bin_vars[headers[1]].tolist()]) == len(bin_vars):
            expression_string = ""
            end_string = ""
            conditions = []
            for _, row in bin_vars.iterrows():
                score = float(row[headers[2]])
                bin_var = row[headers[1]]
                if pd.isnull(bin_var) or "nan" in bin_var:
                    conditions.append((score, f"pandas.isnull(X[0])"))
                else:
                    start, end = extract_elements(bin_var)
                    if start == "-inf":
                        conditions.append((score, f"X[0] < {end}"))
                    elif end == "inf":
                        conditions.append((score, f"X[0] >= {start}"))
                    else:
                        conditions.append((score, f"X[0] >= {start} and X[0] < {end}"))

            if conditions:
                for i, (score, condition) in enumerate(conditions):
                    if i == 0:
                        expression_string += f"{score} if {condition}"
                    elif i == len(conditions) - 1:
                        expression_string += f" else {score}{end_string}"
                    else:
                        expression_string += f" else ({score} if {condition}"
                        end_string += ")"

            mapper.append((
                [var],
                ExpressionTransformer(expression_string),
                # {'alias': var}
            ))
            # samples[var] = np.random.random(n_samples) * 100
        else:
            mapping = {}
            for _, row in bin_vars.iterrows():
                sep = "%,%" if "%,%" in row[headers[1]] else ","
                bin_var = row[headers[1]].split(sep)
                for _bin in bin_var:
                    mapping[_bin] = row[headers[2]]

            mapper.append((
                [var],
                LookupTransformer(mapping=mapping, default_value=0.0),
                # {'alias': var}
            ))
            # samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), n_samples)]

    scorecard_mapper = DataFrameMapper(mapper, df_out=True)

    pipeline = PMMLPipeline([
        ("preprocessing", scorecard_mapper),
        ("scorecard", LinearRegression(fit_intercept=False)),
    ])

    # pipeline.fit(pd.DataFrame(samples), pd.Series(np.random.randint(0, 2, n_samples), name="score"))
    pipeline.named_steps["scorecard"].fit(
        pd.DataFrame(
            np.random.randint(0, 100, (n_samples, len(scorecard_mapper.features))),
            columns=[m[0][0] for m in scorecard_mapper.features]
        ),
        pd.Series(np.random.randint(0, 2, n_samples), name="score")
    )

    pipeline.named_steps["scorecard"].coef_ = np.ones(len(scorecard_mapper.features))

    sklearn2pmml(pipeline, pmml, with_repr=True, debug=debug)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import toad
    import scorecardpy as sc
    from sklearn.linear_model import LogisticRegression

    import warnings

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

    toad_feature_bins = card.export(to_frame=True)

    scorecard_feature_bins = card2bins(card)
    card2pmml(scorecard_feature_bins, pmml="toad_scorecard.pmml")


    # 使用 scorecardpy 生成的评分卡模型转PMML文件
    data_selected = sc.var_filter(data, y=target)
    bins = sc.woebin(data_selected, y=target)
    data_woe = sc.woebin_ply(data_selected, bins)
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
    lr.fit(data_woe.drop(columns=target), data_woe[target])
    card = sc.scorecard(bins, lr, list(data_woe.drop(columns=target).columns))

    scorecardpy_feature_bins = pd.concat({k: v for k, v in card.items() if k != "basepoints"}.values())

    scorecard_feature_bins = card2bins(card)
    card2pmml(scorecard_feature_bins, pmml="scorecardpy_scorecard.pmml")
