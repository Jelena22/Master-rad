import shap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
from build_dataset import split_dataset
import numpy as np
from eda import fill_missing_values, fill_missing_values_actors, encode_categorical_features, tf_idf_transformer


def logistic_regression(merged_df):

    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Primena `fill_missing_values` na trening skup i čuvanje srednje vrednosti trajanja
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

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    logreg = LogisticRegression(
        max_iter=2000,
        random_state=26,
        C=0.1,
        penalty='l2',
        solver='liblinear',
        fit_intercept=False,
        class_weight='balanced'
    )


    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    # PDP
    disp1 = PartialDependenceDisplay.from_estimator(logreg, X_train,
                                                    [1, 0, 46, 47, 48, 49])
    #disp1.plot()
    plt.show()

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

    y_test_np = y_test.values

    for i in [20, 47, 51, 61, 63]:
        print(f"Indeks: {i}")
        print("Detalji:")
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        print(X_test.iloc[i])  # Ispisuje vrednosti funkcija za odabrani indeks
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

    # Pravljenje SHAP explainer-a
    explainer = shap.Explainer(logreg, X_train)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")
    #shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    #shap.force_plot(explainer.expected_value, shap_values, X_test)
    #shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    for i in [20, 47, 51, 61, 63]:
        print(f"Indeks: {i}")
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")
        shap.force_plot(explainer.expected_value, shap_values[i, :], X_test.iloc[i, :], matplotlib=True)

    # print("Expected Value:", explainer.expected_value)
    # print("SHAP Values:", shap_values)

    shap_diff = shap_values[y_test_np != y_pred]
    shap_abs_sum = np.sum(np.abs(shap_diff), axis=0)
    sorted_features = np.argsort(shap_abs_sum)[::-1]
    top_features = X_test.columns[sorted_features]
    print("Najvažnija obeležja:")
    print(top_features)

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(logreg, X_train, y_train, cv=10)

    # Ispis srednje tačnosti i standardne devijacije
    print("Srednja tačnost: {:.2f}%".format(cross_val_scores.mean() * 100))
    print("Standardna devijacija tačnosti: {:.2f}%".format(cross_val_scores.std() * 100))
    for fold, score in enumerate(cross_val_scores, 1):
        print(f"Fold {fold}: Accuracy: {score}")

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("LOGISTIC REGRESSION - Test and predicted data")
    plt.legend()
    plt.show()

def logistic_regression_2(merged_df):
    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    print(y.isna().sum())

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

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    logreg = LogisticRegression(
        max_iter=8000,
        random_state=26,
        C=100.0,
        penalty='l2',
        solver='lbfgs',
        fit_intercept=False,
        class_weight='balanced'
    )

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    # print(y_pred)

    # PDP
    disp1 = PartialDependenceDisplay.from_estimator(logreg, X_train,
                                                    [1, 0, 46, 47, 48, 49])
    #disp1.plot()
    plt.show()

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

    y_test_np = y_test.values

    for i in range(len(y_test_np)):
        if y_test_np[i] != y_pred[i]:
            print(f"Indeks: {i}, Stvarna klasa: {y_test_np[i]}, Predviđena klasa: {y_pred[i]}")

    for i in [20, 47, 51, 61, 63]:
        print(f"Indeks: {i}")
        print("Detalji:")
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        print(X_test.iloc[i])  # Ispisuje vrednosti funkcija za odabrani indeks
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

    # Pravljenje SHAP explainer-a
    explainer = shap.Explainer(logreg, X_train)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type="bar")
    #shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    #shap.force_plot(explainer.expected_value, shap_values, X_test)
    #shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    for i in [2, 3, 7, 27, 39, 43]:
        print(f"Indeks: {i}")
        print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")
        shap.force_plot(explainer.expected_value, shap_values[i, :], X_test.iloc[i, :], matplotlib=True)

    # print("Expected Value:", explainer.expected_value)
    # print("SHAP Values:", shap_values)

    shap_diff = shap_values[y_test_np != y_pred]
    shap_abs_sum = np.sum(np.abs(shap_diff), axis=0)
    sorted_features = np.argsort(shap_abs_sum)[::-1]
    top_features = X_test.columns[sorted_features]
    print("Najvažnija obeležja:")
    print(top_features)

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(logreg, X_train, y_train, cv=10)

    # Ispis srednje tačnosti i standardne devijacije
    print("Srednja tačnost: {:.2f}%".format(cross_val_scores.mean() * 100))
    print("Standardna devijacija tačnosti: {:.2f}%".format(cross_val_scores.std() * 100))
    for fold, score in enumerate(cross_val_scores, 1):
        print(f"Fold {fold}: Accuracy: {score}")

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("LOGISTIC REGRESSION - Test and predicted data")
    plt.legend()
    plt.show()


