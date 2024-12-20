import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from build_dataset import split_dataset
import pandas as pd

from eda import fill_missing_values, fill_missing_values_actors, encode_categorical_features, tf_idf_transformer


def xgboost(merged_df):

    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    X_train, duration_mean = fill_missing_values(X_train)

    # Primena `fill_missing_values` na test skup koristeći srednju vrednost trajanja iz trening skupa
    X_test, _ = fill_missing_values(X_test, duration_mean=duration_mean)

    # Primena TF-IDF samo na kolonu 'Film'
    X_train_tfidf, X_test_tfidf, vectorizer = tf_idf_transformer(X_train['Film'], X_test['Film'])

    # Dodavanje rezultata TF-IDF u DataFrame-ove i uklanjanje originalne kolone 'Film'
    X_train = pd.concat([X_train.drop('Film', axis=1).reset_index(drop=True),
                         pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)
    X_test = pd.concat([X_test.drop('Film', axis=1).reset_index(drop=True),
                        pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)

    # Standardizacija podataka
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    negatives = sum(y_train == 0)
    positives = sum(y_train == 1)
    scale_pos_weight = negatives / positives

    # Kreiranje i treniranje XGBoost modela
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.01,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    # PDP
    disp1 = PartialDependenceDisplay.from_estimator(xgb_model, X_train,
                                                    [1, 0, 46, 47, 48, 49])
    # disp1.plot()
    plt.show()

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=10)

    # Ispis srednje tačnosti i standardne devijacije
    print("Srednja tačnost: {:.2f}%".format(cross_val_scores.mean() * 100))
    print("Standardna devijacija tačnosti: {:.2f}%".format(cross_val_scores.std() * 100))
    for fold, score in enumerate(cross_val_scores, 1):
        print(f"Fold {fold}: Accuracy: {score}")

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Prikaz rezultata evaluacije
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_rep)

    # Dodatne metrike
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Vizualizujte matricu konfuzije
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Dodajte oznake na grafikon
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Dodajte brojeve unutar ćelija
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    if len(y_test) != len(y_pred):
        print("Nizovi y_test i y_pred nemaju istu dužinu.")

    print(f"Indeksi u nizu y_test: {np.unique(y_test)}")

    print("len y_test:", len(y_test))
    print("len y_pred:", len(y_pred))
    print("y_test:", y_test)
    print("y_pred:", y_pred)

    print("Tip y_test:", type(y_test))
    print("Tip y_pred:", type(y_pred))

    y_test_np = y_test.values

    for i in range(len(y_test_np)):
        if y_test_np[i] != y_pred[i]:
            print(f"Indeks: {i}, Stvarna klasa: {y_test_np[i]}, Predviđena klasa: {y_pred[i]}")

    # Pravljenje SHAP explainer-a
    explainer = shap.Explainer(xgb_model, X_train)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    # shap.force_plot(explainer.expected_value, shap_values, X_test)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    for i in [4, 26, 43, 47, 51, 63]:
        print(f"Indeks: {i}")
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")
        shap.force_plot(explainer.expected_value, shap_values[i, :], X_test.iloc[i, :], matplotlib=True)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("XGBOOST - Test and predicted data")
    plt.legend()
    plt.show()


def xgboost_2(merged_df):

    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    X_train, imputer = fill_missing_values_actors(X_train)
    X_test, _ = fill_missing_values_actors(X_test, imputer=imputer)

    # Zatim one-hot enkodiranje
    X_train, encoder = encode_categorical_features(X_train)
    X_test, _ = encode_categorical_features(X_test, encoder=encoder)

    X_train_tfidf, X_test_tfidf, vectorizer = tf_idf_transformer(X_train['Film'], X_test['Film'])

    # Dodavanje rezultata TF-IDF u DataFrame-ove i uklanjanje originalne kolone 'Film'
    X_train = pd.concat([X_train.drop('Film', axis=1).reset_index(drop=True),
                         pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)
    X_test = pd.concat([X_test.drop('Film', axis=1).reset_index(drop=True),
                        pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)

    X_train.columns = X_train.columns.str.replace('[', '(', regex=False)
    X_train.columns = X_train.columns.str.replace(']', ')', regex=False)

    X_test.columns = X_test.columns.str.replace('[', '(', regex=False)
    X_test.columns = X_test.columns.str.replace(']', ')', regex=False)

    # Standardizacija podataka
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    negatives = sum(y_train == 0)
    positives = sum(y_train == 1)
    scale_pos_weight = negatives / positives

    # Kreiranje i treniranje XGBoost modela
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    # PDP
    disp1 = PartialDependenceDisplay.from_estimator(xgb_model, X_train,
                                                    [1, 0, 46, 47, 48, 49])
    # disp1.plot()
    plt.show()

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=10)

    # Ispis srednje tačnosti i standardne devijacije
    print("Srednja tačnost: {:.2f}%".format(cross_val_scores.mean() * 100))
    print("Standardna devijacija tačnosti: {:.2f}%".format(cross_val_scores.std() * 100))
    for fold, score in enumerate(cross_val_scores, 1):
        print(f"Fold {fold}: Accuracy: {score}")

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Prikaz rezultata evaluacije
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_rep)

    # Dodatne metrike
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Vizualizujte matricu konfuzije
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Dodajte oznake na grafikon
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Dodajte brojeve unutar ćelija
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    if len(y_test) != len(y_pred):
        print("Nizovi y_test i y_pred nemaju istu dužinu.")

    print(f"Indeksi u nizu y_test: {np.unique(y_test)}")

    print("len y_test:", len(y_test))
    print("len y_pred:", len(y_pred))
    print("y_test:", y_test)
    print("y_pred:", y_pred)

    print("Tip y_test:", type(y_test))
    print("Tip y_pred:", type(y_pred))

    y_test_np = y_test.values

    for i in range(len(y_test_np)):
        if y_test_np[i] != y_pred[i]:
            print(f"Indeks: {i}, Stvarna klasa: {y_test_np[i]}, Predviđena klasa: {y_pred[i]}")

    print(X_train.dtypes)
    invalid_columns = [col for col in X_train.columns if any(char in col for char in ['[', ']', '<', '>', '\\'])]
    print(invalid_columns)
    print(X_train.shape)
    print(y_train.shape)
    # Pravljenje SHAP explainer-a
    explainer = shap.TreeExplainer(xgb_model)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    # shap.force_plot(explainer.expected_value, shap_values, X_test)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    for i in [2, 7, 8, 27, 43, 52]:
        print(f"Indeks: {i}")
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")
        shap.force_plot(explainer.expected_value, shap_values[i, :], X_test.iloc[i, :], matplotlib=True)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("XGBOOST - Test and predicted data")
    plt.legend()
    plt.show()

