{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tAjYLX7ZUb8W"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score\n",
        "from sklearn.metrics import roc_auc_score\n",
        "from sklearn.metrics import precision_recall_curve\n",
        "from sklearn.metrics import matthews_corrcoef"
      ],
      "metadata": {
        "id": "_v_NBJ7vUfSs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import ExtraTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n",
        "from sklearn.ensemble import AdaBoostClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.ensemble import StackingClassifier\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mHO1iGRlUfa-",
        "outputId": "ea181b56-09ea-4e53-e654-cf2adbf7a211"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/dask/dataframe/__init__.py:42: FutureWarning: \n",
            "Dask dataframe query planning is disabled because dask-expr is not installed.\n",
            "\n",
            "You can install it with `pip install dask[dataframe]` or `conda install dask`.\n",
            "This will raise in a future version.\n",
            "\n",
            "  warnings.warn(msg, FutureWarning)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Train\n",
        "df_tr = pd.read_csv('/content/Re-AAC+APAAC.csv')\n",
        "columns = df_tr.columns.tolist()\n",
        "# Filter the columns to remove data we do not want\n",
        "columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# Store the variable we are predicting\n",
        "target = \"Target\"\n",
        "X = df_tr[columns]\n",
        "Y = df_tr[target]"
      ],
      "metadata": {
        "id": "RTqXmQWsUfeL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# #Test\n",
        "# df_ts = pd.read_csv('/content/Re-AAC+APAAC_Test.csv')\n",
        "# columns = df_ts.columns.tolist()\n",
        "# # Filter the columns to remove data we do not want\n",
        "# columns = [c for c in columns if c not in [\"Target\"]]\n",
        "# # Store the variable we are predicting\n",
        "# target = \"Target\"\n",
        "# xtest = df_ts[columns]\n",
        "# ytest = df_ts[target]"
      ],
      "metadata": {
        "id": "9-vk_9A4UfhI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3)"
      ],
      "metadata": {
        "id": "mzplVHnUUfj0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ensemble = []"
      ],
      "metadata": {
        "id": "yCU1ognxUfnQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model1 = RandomForestClassifier(n_estimators = 400, max_depth = 19)\n",
        "ensemble.append(('RandomForest',model1))"
      ],
      "metadata": {
        "id": "gFNz3SKrUto2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model2 = XGBClassifier(n_estimators = 200,max_depth = 3, learning_rate = 0.22)\n",
        "ensemble.append(('XGBClassifier',model2))"
      ],
      "metadata": {
        "id": "3fA5Cs5MUt27"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model3 = LGBMClassifier(learning_rate = 0.2,max_depth = 15,random_state = 42)\n",
        "ensemble.append(('LGBMClassifier',model3))"
      ],
      "metadata": {
        "id": "nOuudIy6IAOe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model4 = GradientBoostingClassifier(n_estimators = 400, learning_rate = 0.1, random_state = 42)\n",
        "ensemble.append(('GradientBoostingClassifier',model4))"
      ],
      "metadata": {
        "id": "EPO6a6lGIAQn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model5 = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)\n",
        "ensemble.append(('AdaBoost',model5))"
      ],
      "metadata": {
        "id": "-QkmkQ6OIAUE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import VotingClassifier"
      ],
      "metadata": {
        "id": "7LacGIb_IGoe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_ensemble = VotingClassifier(ensemble)"
      ],
      "metadata": {
        "id": "42ukkRnAIGt8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_ensemble.fit(xtrain,ytrain)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "skhd7R7WILHi",
        "outputId": "7b48e561-3c2d-480c-e81a-2ccf471612fa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Info] Number of positive: 311, number of negative: 376\n",
            "[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000675 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 9829\n",
            "[LightGBM] [Info] Number of data points in the train set: 687, number of used features: 44\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.452693 -> initscore=-0.189796\n",
            "[LightGBM] [Info] Start training from score -0.189796\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "VotingClassifier(estimators=[('RandomForest',\n",
              "                              RandomForestClassifier(max_depth=19,\n",
              "                                                     n_estimators=400)),\n",
              "                             ('XGBClassifier',\n",
              "                              XGBClassifier(base_score=None, booster=None,\n",
              "                                            callbacks=None,\n",
              "                                            colsample_bylevel=None,\n",
              "                                            colsample_bynode=None,\n",
              "                                            colsample_bytree=None, device=None,\n",
              "                                            early_stopping_rounds=None,\n",
              "                                            enable_categorical=False,\n",
              "                                            eval_metric=None,\n",
              "                                            feature_types=None, gamma=No...\n",
              "                                            multi_strategy=None,\n",
              "                                            n_estimators=200, n_jobs=None,\n",
              "                                            num_parallel_tree=None,\n",
              "                                            random_state=None, ...)),\n",
              "                             ('LGBMClassifier',\n",
              "                              LGBMClassifier(learning_rate=0.2, max_depth=15,\n",
              "                                             random_state=42)),\n",
              "                             ('GradientBoostingClassifier',\n",
              "                              GradientBoostingClassifier(n_estimators=400,\n",
              "                                                         random_state=42)),\n",
              "                             ('AdaBoost',\n",
              "                              AdaBoostClassifier(learning_rate=0.1,\n",
              "                                                 n_estimators=500,\n",
              "                                                 random_state=42))])"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>VotingClassifier(estimators=[(&#x27;RandomForest&#x27;,\n",
              "                              RandomForestClassifier(max_depth=19,\n",
              "                                                     n_estimators=400)),\n",
              "                             (&#x27;XGBClassifier&#x27;,\n",
              "                              XGBClassifier(base_score=None, booster=None,\n",
              "                                            callbacks=None,\n",
              "                                            colsample_bylevel=None,\n",
              "                                            colsample_bynode=None,\n",
              "                                            colsample_bytree=None, device=None,\n",
              "                                            early_stopping_rounds=None,\n",
              "                                            enable_categorical=False,\n",
              "                                            eval_metric=None,\n",
              "                                            feature_types=None, gamma=No...\n",
              "                                            multi_strategy=None,\n",
              "                                            n_estimators=200, n_jobs=None,\n",
              "                                            num_parallel_tree=None,\n",
              "                                            random_state=None, ...)),\n",
              "                             (&#x27;LGBMClassifier&#x27;,\n",
              "                              LGBMClassifier(learning_rate=0.2, max_depth=15,\n",
              "                                             random_state=42)),\n",
              "                             (&#x27;GradientBoostingClassifier&#x27;,\n",
              "                              GradientBoostingClassifier(n_estimators=400,\n",
              "                                                         random_state=42)),\n",
              "                             (&#x27;AdaBoost&#x27;,\n",
              "                              AdaBoostClassifier(learning_rate=0.1,\n",
              "                                                 n_estimators=500,\n",
              "                                                 random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">VotingClassifier</label><div class=\"sk-toggleable__content\"><pre>VotingClassifier(estimators=[(&#x27;RandomForest&#x27;,\n",
              "                              RandomForestClassifier(max_depth=19,\n",
              "                                                     n_estimators=400)),\n",
              "                             (&#x27;XGBClassifier&#x27;,\n",
              "                              XGBClassifier(base_score=None, booster=None,\n",
              "                                            callbacks=None,\n",
              "                                            colsample_bylevel=None,\n",
              "                                            colsample_bynode=None,\n",
              "                                            colsample_bytree=None, device=None,\n",
              "                                            early_stopping_rounds=None,\n",
              "                                            enable_categorical=False,\n",
              "                                            eval_metric=None,\n",
              "                                            feature_types=None, gamma=No...\n",
              "                                            multi_strategy=None,\n",
              "                                            n_estimators=200, n_jobs=None,\n",
              "                                            num_parallel_tree=None,\n",
              "                                            random_state=None, ...)),\n",
              "                             (&#x27;LGBMClassifier&#x27;,\n",
              "                              LGBMClassifier(learning_rate=0.2, max_depth=15,\n",
              "                                             random_state=42)),\n",
              "                             (&#x27;GradientBoostingClassifier&#x27;,\n",
              "                              GradientBoostingClassifier(n_estimators=400,\n",
              "                                                         random_state=42)),\n",
              "                             (&#x27;AdaBoost&#x27;,\n",
              "                              AdaBoostClassifier(learning_rate=0.1,\n",
              "                                                 n_estimators=500,\n",
              "                                                 random_state=42))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>RandomForest</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(max_depth=19, n_estimators=400)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>XGBClassifier</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBClassifier</label><div class=\"sk-toggleable__content\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
              "              colsample_bylevel=None, colsample_bynode=None,\n",
              "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
              "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
              "              gamma=None, grow_policy=None, importance_type=None,\n",
              "              interaction_constraints=None, learning_rate=0.22, max_bin=None,\n",
              "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
              "              max_delta_step=None, max_depth=3, max_leaves=None,\n",
              "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
              "              multi_strategy=None, n_estimators=200, n_jobs=None,\n",
              "              num_parallel_tree=None, random_state=None, ...)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>LGBMClassifier</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LGBMClassifier</label><div class=\"sk-toggleable__content\"><pre>LGBMClassifier(learning_rate=0.2, max_depth=15, random_state=42)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>GradientBoostingClassifier</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GradientBoostingClassifier</label><div class=\"sk-toggleable__content\"><pre>GradientBoostingClassifier(n_estimators=400, random_state=42)</pre></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><label>AdaBoost</label></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">AdaBoostClassifier</label><div class=\"sk-toggleable__content\"><pre>AdaBoostClassifier(learning_rate=0.1, n_estimators=500, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "new_ensemble.score(xtest,ytest)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p2tW57m_ILSV",
        "outputId": "ebb6ca3c-107f-41c5-ab3a-678035652345"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9796610169491525"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics = []\n",
        "total_Metics = pd.DataFrame(total_Metics)\n",
        "total_Metics['Classifier'] = 'Classifier'\n",
        "total_Metics['mcc'] = 'mcc'\n",
        "# total_Metics['auc'] = 'auc'\n",
        "total_Metics['Kappa'] = 'Kappa'\n",
        "total_Metics['precision'] = 'precision'\n",
        "total_Metics['recall'] = 'recall'\n",
        "total_Metics['f1'] = 'f1'\n",
        "total_Metics['sensitivity'] = 'sensitivity'\n",
        "total_Metics['specificity'] = 'specificity'\n",
        "\n",
        "cv = KFold(n_splits=10, random_state=1, shuffle=True)"
      ],
      "metadata": {
        "id": "Uvzb0-_KXt0m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "total_Metics2 = []\n",
        "total_Metics2 = pd.DataFrame(total_Metics2)\n",
        "total_Metics2['Classifier'] = 'Classifier'\n",
        "total_Metics2['mcc'] = 'mcc'\n",
        "# total_Metics2['auc'] = 'auc'\n",
        "total_Metics2['Kappa'] = 'Kappa'\n",
        "total_Metics2['precision'] = 'precision'\n",
        "total_Metics2['recall'] = 'recall'\n",
        "total_Metics2['f1'] = 'f1'\n",
        "total_Metics2['sensitivity'] = 'sensitivity'\n",
        "total_Metics2['specificity'] = 'specificity'"
      ],
      "metadata": {
        "id": "qqZuB-ruX4jX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "pred = cross_val_predict(new_ensemble, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "mcc = matthews_corrcoef(ytrain, pred)\n",
        "cm1 = confusion_matrix(ytrain, pred)\n",
        "kappa = cohen_kappa_score(ytrain, pred)\n",
        "f1 = f1_score(ytrain, pred)\n",
        "precision_score = precision_score(ytrain, pred)\n",
        "recall_score = recall_score(ytrain, pred)\n",
        "sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "total_Metics.loc[len(total_Metics.index)] = [new_ensemble, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]"
      ],
      "metadata": {
        "id": "Ew8FG58lX4px"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score\n",
        "new_ensemble.fit(xtrain,ytrain)\n",
        "pred = new_ensemble.predict(xtest)\n",
        "mcc = matthews_corrcoef(ytest, pred)\n",
        "cm1 = confusion_matrix(ytest, pred)\n",
        "kappa = cohen_kappa_score(ytest, pred)\n",
        "f1 = f1_score(ytest, pred)\n",
        "precision_score = precision_score(ytest, pred)\n",
        "recall_score = recall_score(ytest, pred)\n",
        "sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])\n",
        "specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])\n",
        "total_Metics2.loc[len(total_Metics2.index)] = [new_ensemble, mcc, kappa, precision_score,recall_score, f1, sensitivity,specificity]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4eZ5LifhX4sQ",
        "outputId": "75b5c4f5-01da-4284-e39c-ff90c53365ff"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[LightGBM] [Info] Number of positive: 311, number of negative: 376\n",
            "[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000655 seconds.\n",
            "You can set `force_col_wise=true` to remove the overhead.\n",
            "[LightGBM] [Info] Total Bins 9829\n",
            "[LightGBM] [Info] Number of data points in the train set: 687, number of used features: 44\n",
            "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.452693 -> initscore=-0.189796\n",
            "[LightGBM] [Info] Start training from score -0.189796\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n",
            "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(total_Metics)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7AZiHQZsX4vk",
        "outputId": "4acf4b6e-9353-4042-e9e6-27dd8f95850d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier      mcc     Kappa  \\\n",
            "0  VotingClassifier(estimators=[('RandomForest',\\...  0.85976  0.859391   \n",
            "\n",
            "   precision    recall        f1  sensitivity  specificity  \n",
            "0   0.909657  0.938907  0.924051     0.922872     0.938907  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(total_Metics2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1wpP7EmZYGEv",
        "outputId": "93d8b14b-48db-4eff-adc7-0d7a7b24d508"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                                          Classifier       mcc     Kappa  \\\n",
            "0  VotingClassifier(estimators=[('RandomForest',\\...  0.945461  0.945373   \n",
            "\n",
            "   precision    recall       f1  sensitivity  specificity  \n",
            "0   0.977612  0.963235  0.97037     0.981132     0.963235  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "h_pred = cross_val_predict(new_ensemble, xtrain, ytrain, cv=cv, n_jobs=-1)\n",
        "h_Accuracy = accuracy_score(ytrain, h_pred)"
      ],
      "metadata": {
        "id": "RyuvV3tPXeeA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "h_Accuracy"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MHkc-da-XegF",
        "outputId": "a90f2f84-5868-4efe-a741-4708497736ee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9301310043668122"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "gKVICIg7XeiF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "BMLsULOEXelv"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
