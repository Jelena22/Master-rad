import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from build_dataset import split_dataset
import pandas as pd

from eda import fill_missing_values, fill_missing_values_actors, encode_categorical_features, tf_idf_transformer


def svm(merged_df):
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

    # Čuvanje imena kolona za SHAP analizu
    feature_names = X_train.columns

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_test shape:", y_test.shape)

    # Kreiranje i treniranje SVM modela
    svm_model = SVC(random_state=42)
    svm_model.feature_names_in_ = list(X.columns)
    svm_model.fit(X_train_scaled, y_train)

    # # SHAP values
    # explainer = shap.Explainer(svm_model.predict, X_train_scaled)
    # # Calculates the SHAP values - It takes some time
    # shap_values = explainer(X_test_scaled)
    # # Evaluate SHAP values
    # #shap.plots.bar(shap_values)
    # shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar")

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid'],
                   'degree': [2, 3, 4],
                   'coef0': [0.0, 0.1, 0.5]}

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Grid Search Cross Validation sa 10-fold
    grid_search = GridSearchCV(svm_model, param_grid, cv=kf)
    grid_search.fit(X_train_scaled, y_train)

    # Ispis najboljih parametara
    print("Najbolji parametri:", grid_search.best_params_)

    # Predviđanje na test skupu
    y_pred = grid_search.predict(X_test_scaled)

    # Kros-validacione tačnosti
    cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=kf)

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

    print("Broj kolona u X_test_scaled:", X_test_scaled.shape[1])
    print("Broj kolona u X:", X.shape[1])

    # SHAP values
    explainer = shap.Explainer(svm_model.decision_function, X_train_scaled)
    # Izračunavanje SHAP vrednosti
    shap_values = explainer(X_test_scaled)
    # Prikaz SHAP vrednosti
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar")

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
    plt.title("SVM - Test and predicted data")
    plt.legend()
    plt.show()

def svm_2(merged_df):

    X = merged_df.drop(['Winner'], axis=1)
    y = merged_df['Winner']

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    X_train, imputer = fill_missing_values_actors(X_train)
    X_test, _ = fill_missing_values_actors(X_test, imputer=imputer)

    #one-hot enkodiranje
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

    svm_model = SVC(probability=True, random_state=42)
    param_dist = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'sigmoid']
                  }

    # Grid Search with 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(svm_model, param_dist, n_iter=10, cv=kf, random_state=42)
    random_search.fit(X_train_scaled, y_train)

    # Najbolji parametri
    print("Najbolji parametri:", random_search.best_params_)

    # Predikcije na test skupu
    y_pred_svm = random_search.predict(X_test_scaled)

    # Cross-validation scores
    cross_val_scores = cross_val_score(random_search.best_estimator_, X_train_scaled, y, cv=kf)
    print("Srednja tačnost: {:.2f}%".format(cross_val_scores.mean() * 100))
    print("Standardna devijacija tačnosti: {:.2f}%".format(cross_val_scores.std() * 100))
    for fold, score in enumerate(cross_val_scores, 1):
        print(f"Fold {fold}: Accuracy: {score}")

    # Compute and print overall metrics on the test set
    accuracy = accuracy_score(y_test, y_pred_svm)
    conf_matrix = confusion_matrix(y_test, y_pred_svm)
    class_report = classification_report(y_test, y_pred_svm)

    print("\nOverall Accuracy: {:.2f}%".format(accuracy * 100))
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Dodatne metrike
    precision = precision_score(y_test, y_pred_svm)
    recall = recall_score(y_test, y_pred_svm)
    f1 = f1_score(y_test, y_pred_svm)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Summarize background data
    K = 100  # Choose a suitable number of background samples
    X_train_summary = shap.sample(X_train_scaled, K)  # or use shap.kmeans(X_train, K)

    # Compute SHAP values
    explainer = shap.KernelExplainer(random_search.best_estimator_.predict_proba, X_train_summary)
    shap_values = explainer.shap_values(X_test_scaled[:100])  # Use a subset of your test data for SHAP

    # Plot SHAP values (only for binary classification example here)
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X_train_scaled.columns[:-1])
