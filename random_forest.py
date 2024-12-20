import numpy as np
import shap
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from build_dataset import split_dataset
import pandas as pd

from eda import fill_missing_values, fill_missing_values_actors, encode_categorical_features, tf_idf_transformer


def random_forest(merged_df):
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

    # Kreiranje RandomForestClassifier modela
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=False,
        random_state=42,
    )

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=10)

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

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("RANDOM FOREST - Test and predicted data")
    plt.legend()
    plt.show()

    # Pravljenje SHAP explainer-a
    explainer = shap.TreeExplainer(rf_classifier, X_train)

    # # Postavljanje prosečnih vrednosti ciljne promenljive kao očekivanu vrednost
    # expected_values = np.mean(y_train)
    # explainer.expected_value = expected_values

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values[1], X_test, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    for i in [26, 43, 47, 51, 63]:
        try:
            i = int(i)
            if 0 <= i < len(X_test) and i < len(shap_values[0]):
                print(f"Indeks: {i}")
                print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

                # Access the i-th output for each feature
                shap_values_i = np.array([values[i] for values in shap_values])
                # shap_values = explainer.shap_values(i)
                # shap.initjs()
                #shap_values_i = np.array(explainer.shap_values(X_test.iloc[i, :]))

                # Use shap.plots.force instead of shap.force_plot
                shap.plots.force(explainer.expected_value[1], shap_values_i[1], X_test.iloc[i, :], matplotlib=True)
            else:
                print(
                    f"Index {i} is out of range. Max index for X_test: {len(X_test) - 1}, Max index for shap_values: {len(shap_values[0]) - 1}")
        except Exception as e:
            print(f"Error processing index {i}: {e}")


def random_forest_bagging(merged_df):
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_classifier = RandomForestClassifier(
        n_estimators=10,
        criterion='gini',
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        random_state=42
    )

    # Kreiranje BaggingClassifier-a sa RandomForestClassifier-om kao baznim modelom
    bagged_rf_classifier = BaggingClassifier(estimator=rf_classifier, n_estimators=10, random_state=42)

    bagged_rf_classifier.fit(X_train_scaled, y_train)
    y_pred = bagged_rf_classifier.predict(X_test)

    # Primena 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(bagged_rf_classifier, X_train_scaled, y_train, cv=kf)

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
    explainer = shap.TreeExplainer(bagged_rf_classifier.estimators_[0], X_train_scaled)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test_scaled)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    print("Length of y_test:", len(y_test))
    print("Length of y_pred:", len(y_pred))

    print("y_test:", y_test)
    print("y_pred:", y_pred)

    for i in [26, 43, 47, 51, 63]:
        try:
            print(f"Index: {i}")
            print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

            # Access the i-th output for each feature
            shap_values_i = np.array([values[i] for values in shap_values])

            # Use shap.plots.force instead of shap.force_plot
            shap.plots.force(explainer.expected_value[1], shap_values_i[1], X_test_scaled[i], matplotlib=True, feature_names=X.columns)
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("RANDOM FOREST BAGGING - Test and predicted data")
    plt.legend()
    plt.show()

def random_forest_2(merged_df):
    #merged_df = tf_idf(merged_df)

    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

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

    # Kreiranje RandomForestClassifier modela
    rf_classifier = RandomForestClassifier(
        n_estimators=300,
        criterion='entropy',
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.6,
        bootstrap=True,
        random_state=42,
        class_weight='balanced'

    )

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    # Primena 10-fold cross-validation na trening skupu
    cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=10)

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

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("RANDOM FOREST - Test and predicted data")
    plt.legend()
    plt.show()

    explainer = shap.TreeExplainer(rf_classifier)

    # Izračunavanje SHAP vrednosti za X_test
    shap_values = explainer.shap_values(X_test)

    # Ako radiš sa binarnom klasifikacijom
    shap.summary_plot(shap_values[1], X_test, feature_names=X_train.columns, plot_type="bar")

    # Prikazivanje SHAP force plot-a za specifičan indeks
    for i in [2, 7, 8, 27, 29]:
        try:
            i = int(i)
            if 0 <= i < len(X_test) and i < len(shap_values[0]):
                print(f"Indeks: {i}")
                print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

                # Access the i-th output for each feature
                shap_values_i = np.array([values[i] for values in shap_values])
                # shap_values = explainer.shap_values(i)
                # shap.initjs()
                #shap_values_i = np.array(explainer.shap_values(X_test.iloc[i, :]))

                # Use shap.plots.force instead of shap.force_plot
                shap.plots.force(explainer.expected_value[1], shap_values_i[1], X_test.iloc[i, :], matplotlib=True)
            else:
                print(
                    f"Index {i} is out of range. Max index for X_test: {len(X_test) - 1}, Max index for shap_values: {len(shap_values[0]) - 1}")
        except Exception as e:
            print(f"Error processing index {i}: {e}")


def random_forest_bagging_2(merged_df):
    #merged_df = tf_idf(merged_df)

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    brf_classifier = BalancedRandomForestClassifier(
        n_estimators=200,
        random_state=42,
        criterion='entropy',
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.6,
        bootstrap=True,
        sampling_strategy='auto'
    )

    # Kreiranje BaggingClassifier-a sa RandomForestClassifier-om kao baznim modelom
    bagged_rf_classifier = BaggingClassifier(estimator=brf_classifier, n_estimators=10, random_state=42)

    bagged_rf_classifier.fit(X_train_scaled, y_train)
    y_pred = bagged_rf_classifier.predict(X_test)

    # Primena 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(bagged_rf_classifier, X_train_scaled, y_train, cv=kf)

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
    explainer = shap.TreeExplainer(bagged_rf_classifier.estimators_[0], X_train_scaled)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test_scaled)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # Prikazivanje SHAP force plot-a za specifičan indeks
    # shap.force_plot(explainer.expected_value, shap_values, X_test)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])

    print("Length of y_test:", len(y_test))
    print("Length of y_pred:", len(y_pred))

    print("y_test:", y_test)
    print("y_pred:", y_pred)

    for i in [26, 43, 47, 51, 63]:
        try:
            print(f"Index: {i}")
            print(f"Stvarna klasa: {y_test.iloc[i]}, Predviđena klasa: {y_pred[i]}\n")

            # Access the i-th output for each feature
            shap_values_i = np.array([values[i] for values in shap_values])

            # Use shap.plots.force instead of shap.force_plot
            shap.plots.force(explainer.expected_value[1], shap_values_i[1], X_test_scaled[i], matplotlib=True, feature_names=X_train.columns)
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("RANDOM FOREST BAGGING - Test and predicted data")
    plt.legend()
    plt.show()