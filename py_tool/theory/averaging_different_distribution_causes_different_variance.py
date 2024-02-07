import numpy as np

model_count = 8

all_models = []
for i in range(model_count):
    # data = np.random.uniform(low=-0.5, high=0.5, size=10000)
    # [0.0836762123895337, 0.08423390119638642, 0.08260949944087236, 0.08198181271152918, 0.08292232455113609, 0.0827220247175202, 0.08327187093528192, 0.08338497277543955]
    # 0.010498424942142972

    data = np.random.normal(loc=0, scale=0.5, size=10000)
    # [0.25302686443775446, 0.24180093486796395, 0.25495720937288524, 0.24807832126547683, 0.2516332727663104, 0.24734904022002685, 0.24736056959149313, 0.24845968709139837]
    # 0.03102792351815493

    # data = np.random.binomial(1, 0.5, size=10000)
    # [0.24980119000000003, 0.24997296000000002, 0.24998151, 0.24999099999999994, 0.24999158999999999, 0.24997399000000006, 0.24997191000000002, 0.24999711000000002]
    # 0.031149984375

    # data = np.random.dirichlet([10], size=10000)
    # [1.756448109281159e-33, 1.7613784899387904e-33, 1.7022139220472145e-33, 1.763843680267606e-33, 1.765076275432014e-33, 1.713307278526885e-33, 1.765076275432014e-33, 1.7022139220472145e-33]
    # 3.6977854932234925e-36

    all_models.append(data)

vars = []
output_model = np.zeros(all_models[0].shape)
for m in all_models:
    output_model = output_model + m
    vars.append(np.var(m))
output_model = output_model/model_count
output_var = np.var(output_model)
print(vars)
print(output_var)