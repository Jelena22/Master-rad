import numpy as np
import pandas as pd
import shap
from keras.src.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout
from tensorflow.python.keras.regularizers import l2

from build_dataset import split_dataset
from eda import fill_missing_values, fill_missing_values_actors, encode_categorical_features, tf_idf_transformer


def ann(merged_df):

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

    print("Tip podataka X_train_scaled:", type(X_train_scaled))
    print("Oblik X_train_scaled:", X_train_scaled.shape)

    print("Tip podataka y_train:", type(y_train))
    print("Oblik y_train:", y_train.shape)

    print(np.unique(y_train))

    # # Inicijalizacija 10-fold cross-validation
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #
    # # Lista za čuvanje tačnosti za svaki fold
    # accuracies = []
    #
    # # Iteracija kroz foldove
    # for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    #     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    #
    #     # Skaliranje podataka
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
    #
    #     # Kreiranje modela neuronske mreže
    #     model = Sequential()
    #     model.add(Input(shape=(X_train_scaled.shape[1],)))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     # Kompilacija modela
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    #     # Treniranje modela
    #     model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, verbose=0)
    #
    #     # Evaluacija na test skupu
    #     y_pred_probs = model.predict(X_test_scaled)
    #     y_pred = (y_pred_probs > 0.5).astype(int)
    #
    #     # Evaluacija tačnosti za trenutni fold
    #     accuracy = accuracy_score(y_test, y_pred)
    #     accuracies.append(accuracy)
    #
    #     # Ispisivanje rezultata za trenutni fold
    #     print(f"Fold {fold_idx + 1} - Accuracy: {accuracy * 100:.2f}%")
    #     print("\nFold {}:".format(fold_idx + 1))
    #     print("Confusion Matrix:")
    #     print(confusion_matrix(y_test, y_pred))
    #     print("Classification Report:")
    #     print(classification_report(y_test, y_pred))
    #
    # # Ispisivanje srednje tačnosti preko svih foldova
    # mean_accuracy = np.mean(accuracies)
    # print(f"\nSrednja tačnost: {mean_accuracy * 100:.2f}%")

    # Kreiranje modela neuronske mreže
    model = Sequential()
    model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Za binarnu klasifikaciju

    # Kompilacija modela
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treniranje modela
    history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=15, validation_data=(X_test_scaled, y_test), verbose=1)

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluacija na test skupu
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Evaluacija tačnosti
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy: {:.2f}%".format(accuracy * 100))

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
    explainer = shap.Explainer(model, X_train_scaled)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test_scaled)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)

def ann_2(merged_df):
    # print(df_encoded)
    # merged_df = tf_idf(merged_df)

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

    print("Tip podataka X_train_scaled:", type(X_train_scaled))
    print("Oblik X_train_scaled:", X_train_scaled.shape)

    print("Tip podataka y_train:", type(y_train))
    print("Oblik y_train:", y_train.shape)

    print(np.unique(y_train))

    # # Inicijalizacija 10-fold cross-validation
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #
    # # Lista za čuvanje tačnosti za svaki fold
    # accuracies = []
    #
    # # Iteracija kroz foldove
    # for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    #     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    #
    #     # Skaliranje podataka
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
    #
    #     # Kreiranje modela neuronske mreže
    #     model = Sequential()
    #     model.add(Dense(512, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     # Kompilacija modela
    #     model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    #
    #     # Treniranje modela
    #     model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, verbose=1)
    #
    #     # Evaluacija na test skupu
    #     y_pred_probs = model.predict(X_test_scaled)
    #     y_pred = (y_pred_probs > 0.5).astype(int)
    #
    #     # Evaluacija tačnosti za trenutni fold
    #     accuracy = accuracy_score(y_test, y_pred)
    #     accuracies.append(accuracy)
    #
    #     # Ispisivanje rezultata za trenutni fold
    #     print(f"Fold {fold_idx + 1} - Accuracy: {accuracy * 100:.2f}%")
    #     print("\nFold {}:".format(fold_idx + 1))
    #     print("Confusion Matrix:")
    #     print(confusion_matrix(y_test, y_pred))
    #     print("Classification Report:")
    #     print(classification_report(y_test, y_pred))
    #
    # # Ispisivanje srednje tačnosti preko svih foldova
    # mean_accuracy = np.mean(accuracies)
    # print(f"\nSrednja tačnost: {mean_accuracy * 100:.2f}%")

    # Kreiranje modela neuronske mreže
    model = Sequential()
    model.add(Dense(512, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # Kompilacija modela
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Treniranje modela
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=1)

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluacija na test skupu
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Evaluacija tačnosti
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy: {:.2f}%".format(accuracy * 100))

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
    explainer = shap.Explainer(model, X_train_scaled)

    # Izračunavanje SHAP vrednosti za testni skup
    shap_values = explainer.shap_values(X_test_scaled)

    # Prikazivanje SHAP summary plot-a
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, plot_type="bar")
    # shap.summary_plot(shap_values, X_test, feature_names=X.columns)
