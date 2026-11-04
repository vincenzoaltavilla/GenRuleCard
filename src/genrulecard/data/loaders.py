from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..core.artifacts import ProcessedDataset
from ..core.config import DatasetConfig, SplitConfig


def _as_python_scalar(value):
    """Converte scalari NumPy in valori comodi per CSV/figure/JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _build_onehot_mapping(ohe: OneHotEncoder, categorical_features: list[str]) -> dict[str, dict]:
    """Mappa una colonna OHE alla feature/categoria originaria.

    Serve alla visualizzazione leggibile della scorecard: ad esempio
    ``sex_male > 0.5`` diventa ``sex = 'male'``.
    """
    mapping: dict[str, dict] = {}
    for feature_name, categories in zip(categorical_features, ohe.categories_):
        categories_list = [_as_python_scalar(category) for category in categories]
        for category in categories_list:
            mapping[f"{feature_name}_{category}"] = {
                "original_feature": feature_name,
                "category": category,
                "all_categories": categories_list,
            }
    return mapping


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _titanic_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    path = _project_root() / "data" / "raw" / "titanic.csv"
    if not path.exists():
        raise FileNotFoundError(f"Titanic non trovato: {path}")
    raw = pd.read_csv(path)
    raw.columns = [str(column).lower() for column in raw.columns]

    # Replica le feature semantiche del dataset seaborn usato nel notebook.
    raw["pclass"] = raw["pclass"]
    raw["sex"] = raw["sex"].astype("object")
    raw["embarked"] = raw["embarked"].astype("object")
    raw["class"] = raw["pclass"].map({1: "First", 2: "Second", 3: "Third"})
    raw["who"] = np.where(
        raw["age"].fillna(99) < 18,
        "child",
        np.where(raw["sex"].eq("male"), "man", "woman"),
    )
    raw["adult_male"] = (raw["sex"].eq("male") & raw["age"].fillna(-1).ge(18)).astype(int)
    raw["alone"] = ((raw["sibsp"].fillna(0) + raw["parch"].fillna(0)) == 0).astype(int)
    raw["survived"] = raw["survived"].astype(int)

    cfg = DatasetConfig(
        name="titanic",
        target_column="survived",
        numeric_features=["pclass", "age", "sibsp", "parch", "fare"],
        categorical_features=["sex", "embarked", "class", "who"],
        boolean_features=["adult_male", "alone"],
        positive_class=1,
    )
    return raw, cfg


def _breast_cancer_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    dataset = load_breast_cancer()
    feature_names = [str(name).replace(" ", "_") for name in dataset.feature_names]
    raw = pd.DataFrame(dataset.data, columns=feature_names)
    raw["diagnosis"] = dataset.target.astype(int)

    cfg = DatasetConfig(
        name="breast",
        target_column="diagnosis",
        numeric_features=feature_names,
        categorical_features=[],
        boolean_features=[],
        # sklearn codifica malignant=0 e benign=1: la classe positiva e' il tumore maligno.
        positive_class=0,
    )
    return raw, cfg


def _openml_frame(data_id: int) -> pd.DataFrame:
    dataset = fetch_openml(
        data_id=data_id,
        as_frame=True,
        parser="auto",
        data_home=_project_root() / "data" / "openml",
    )
    if dataset.frame is None:
        raise ValueError(f"OpenML data_id={data_id} non ha restituito un DataFrame.")
    frame = dataset.frame.copy()
    frame.columns = [str(column).strip().replace(" ", "_").replace("-", "_") for column in frame.columns]
    return frame


def _openml_binary_frame(
    *,
    name: str,
    data_id: int,
    positive_class,
    target_candidates: tuple[str, ...] = (),
    categorical_overrides: tuple[str, ...] = (),
    numeric_overrides: tuple[str, ...] = (),
    drop: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, DatasetConfig]:
    """Carica una versione OpenML fissata e ne dichiara il preprocessing.

    Gli ID, invece dei soli nomi OpenML, rendono il benchmark riproducibile e
    impediscono che una nuova versione omonima cambi silenziosamente i dati.
    """
    raw = _openml_frame(data_id)
    raw = raw.drop(columns=[column for column in drop if column in raw.columns])
    target = next((column for column in target_candidates if column in raw.columns), raw.columns[-1])
    features = [column for column in raw.columns if column != target]
    categorical = [column for column in categorical_overrides if column in features]
    numeric = [column for column in numeric_overrides if column in features]
    assigned = set(categorical + numeric)
    for column in features:
        if column in assigned:
            continue
        if pd.api.types.is_numeric_dtype(raw[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw, DatasetConfig(
        name=name,
        target_column=target,
        numeric_features=numeric,
        categorical_features=categorical,
        boolean_features=[],
        positive_class=positive_class,
    )


def _adult_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    raw, config = _openml_binary_frame(
        name="adult", data_id=1590, positive_class=">50K", target_candidates=("class",)
    )
    target = config.target_column
    if target is None:
        raise RuntimeError("Adult non ha una colonna target configurata.")
    raw[target] = raw[target].astype("string").str.strip()
    return raw, config


def _bank_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    # Give Me Some Credit: 150000 righe, 10 predittori, default entro due anni.
    return _openml_binary_frame(
        name="bank", data_id=45577, positive_class="1",
        target_candidates=("SeriousDlqin2yrs",),
    )


def _bank_marketing_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    # UCI Bank Marketing, versione OpenML fissata; Class=2 corrisponde a yes.
    return _openml_binary_frame(
        name="bankMarketing", data_id=1461, positive_class="2",
        target_candidates=("Class", "class"),
    )


def _compas_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    return _openml_binary_frame(
        name="compas", data_id=42192, positive_class="1",
        target_candidates=("two_year_recid",),
    )


def _magic_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    return _openml_binary_frame(
        name="magic", data_id=1120, positive_class="g",
        target_candidates=("class_", "class"),
    )


def _mushroom_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    return _openml_binary_frame(
        name="mushroom", data_id=24, positive_class="p", target_candidates=("class",)
    )


def _taiwan_credit_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    # Manteniamo i 23 attributi numerici della formulazione benchmark originale.
    return _openml_binary_frame(
        name="taiwan_credit", data_id=42477, positive_class="1", target_candidates=("y",)
    )


def _telco_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    raw = _openml_frame(45568)
    if "TotalCharges" in raw.columns:
        raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    target = "Churn"
    features = [column for column in raw.columns if column != target]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(raw[column])]
    categorical = [column for column in features if column not in numeric]
    return raw, DatasetConfig(
        name="telco", target_column=target, numeric_features=numeric,
        categorical_features=categorical, boolean_features=[], positive_class="Yes",
    )


def _diabetes_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    """Pima Indians Diabetes, OpenML data_id=37: 768 righe, 8 feature numeriche."""
    raw = _openml_frame(37)
    target = "class"
    if target not in raw.columns:
        target = raw.columns[-1]
    numeric = [column for column in raw.columns if column != target]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    cfg = DatasetConfig(
        name="diabetes",
        target_column=target,
        numeric_features=numeric,
        categorical_features=[],
        boolean_features=[],
        positive_class="tested_positive",
    )
    return raw, cfg


def _german_credit_frame() -> tuple[pd.DataFrame, DatasetConfig]:
    """German Credit, OpenML data_id=31: 1000 righe, target good/bad."""
    raw = _openml_frame(31)
    target = "class"
    if target not in raw.columns:
        target = raw.columns[-1]
    feature_columns = [column for column in raw.columns if column != target]
    numeric = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(raw[column])
    ]
    categorical = [column for column in feature_columns if column not in numeric]

    cfg = DatasetConfig(
        name="german_credit",
        target_column=target,
        numeric_features=numeric,
        categorical_features=categorical,
        boolean_features=[],
        positive_class="bad",
    )
    return raw, cfg


def _custom_frame(config: DatasetConfig) -> tuple[pd.DataFrame, DatasetConfig]:
    if config.csv_path is None or config.target_column is None:
        raise ValueError("Per name='csv' specifica csv_path e target_column in main.py.")
    path = Path(config.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV non trovato: {path}")
    frame = pd.read_csv(path)
    if config.target_column not in frame.columns:
        raise ValueError(f"Target '{config.target_column}' non presente nel CSV.")
    if not config.numeric_features and not config.categorical_features and not config.boolean_features:
        raise ValueError("Per name='csv' specifica almeno una lista di feature.")
    return frame, config


def _build_design_matrix(
    frame: pd.DataFrame,
    numerical: list[str],
    boolean: list[str],
    encoded_categories: np.ndarray,
    category_names: list[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            frame[numerical + boolean].reset_index(drop=True),
            pd.DataFrame(encoded_categories, columns=category_names),
        ],
        axis=1,
    )


def load_and_split_dataset(config: DatasetConfig, split: SplitConfig) -> ProcessedDataset:
    dataset_name = config.name.lower()
    if dataset_name == "adult":
        raw, active_config = _adult_frame()
    elif dataset_name == "bank":
        raw, active_config = _bank_frame()
    elif dataset_name in {"bankmarketing", "bank_marketing"}:
        raw, active_config = _bank_marketing_frame()
    elif dataset_name in {"breast", "breast_cancer"}:
        raw, active_config = _breast_cancer_frame()
    elif dataset_name in {"compas", "compass"}:
        raw, active_config = _compas_frame()
    elif dataset_name == "titanic":
        raw, active_config = _titanic_frame()
    elif dataset_name == "diabetes":
        raw, active_config = _diabetes_frame()
    elif dataset_name in {"german", "german_credit", "credit_g"}:
        raw, active_config = _german_credit_frame()
    elif dataset_name == "magic":
        raw, active_config = _magic_frame()
    elif dataset_name == "mushroom":
        raw, active_config = _mushroom_frame()
    elif dataset_name in {"taiwan", "taiwan_credit"}:
        raw, active_config = _taiwan_credit_frame()
    elif dataset_name == "telco":
        raw, active_config = _telco_frame()
    elif dataset_name == "csv":
        raw, active_config = _custom_frame(config)
    else:
        raise ValueError(
            "Dataset supportati: adult, bank, bankMarketing, breast, compas, diabetes, "
            "german_credit, magic, mushroom, taiwan_credit, telco, titanic e csv."
        )

    target = active_config.target_column
    numeric = list(active_config.numeric_features or [])
    categorical = list(active_config.categorical_features or [])
    boolean = list(active_config.boolean_features or [])
    selected = numeric + categorical + boolean
    missing = set(selected + [target]).difference(raw.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing)}")

    X_raw = raw[selected].copy()
    y_raw = raw[target].copy()
    classes = np.sort(pd.unique(y_raw))
    if len(classes) != 2:
        raise ValueError("GenRuleCard supporta al momento solo classificazione binaria.")
    if active_config.positive_class not in set(classes):
        raise ValueError("positive_class non presente nel target.")
    # Il notebook assume classi [0, 1]; per dataset generici normalizziamo
    # rispettando positive_class come classe 1.
    negative_class = classes[0] if classes[1] == active_config.positive_class else classes[1]
    mapping = {negative_class: 0, active_config.positive_class: 1}
    y_raw = y_raw.map(mapping).astype(int)

    # Ricrea una sola volta il test storico e lo mantiene fisso. Il precedente
    # train e la precedente validation formano il development set, che può poi
    # essere risuddiviso con un seed indipendente.
    holdout_size = split.validation_size + split.test_size
    X_base_train, X_holdout, y_base_train, y_holdout = train_test_split(
        X_raw, y_raw, test_size=holdout_size,
        random_state=split.test_random_state, stratify=y_raw,
    )
    validation_share_of_holdout = split.validation_size / holdout_size
    X_base_val, X_test_raw, y_base_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=1.0 - validation_share_of_holdout,
        random_state=split.test_random_state,
        stratify=y_holdout,
    )
    X_development = pd.concat([X_base_train, X_base_val], axis=0)
    y_development = pd.concat([y_base_train, y_base_val], axis=0)
    validation_share = split.validation_size / (split.train_size + split.validation_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=validation_share,
        random_state=split.validation_random_state,
        stratify=y_development,
    )
    # Tutte le statistiche di preprocessing sono apprese solo dal train.
    for col in numeric:
        fill = X_train_raw[col].median()
        for frame in (X_train_raw, X_val_raw, X_test_raw):
            frame.loc[:, col] = pd.to_numeric(frame[col], errors="coerce").fillna(fill)
    for col in categorical:
        mode = X_train_raw[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "<missing>"
        for frame in (X_train_raw, X_val_raw, X_test_raw):
            frame.loc[:, col] = frame[col].astype("object").fillna(fill)
    for col in boolean:
        for frame in (X_train_raw, X_val_raw, X_test_raw):
            frame[col] = frame[col].astype("boolean").fillna(False).astype("int8")

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if categorical:
        cat_train = ohe.fit_transform(X_train_raw[categorical])
        cat_val = ohe.transform(X_val_raw[categorical])
        cat_test = ohe.transform(X_test_raw[categorical])
        cat_names = list(ohe.get_feature_names_out(categorical))
    else:
        cat_train = np.empty((len(X_train_raw), 0), dtype=np.float32)
        cat_val = np.empty((len(X_val_raw), 0), dtype=np.float32)
        cat_test = np.empty((len(X_test_raw), 0), dtype=np.float32)
        cat_names = []

    train_df = _build_design_matrix(X_train_raw, numeric, boolean, cat_train, cat_names)
    val_df = _build_design_matrix(X_val_raw, numeric, boolean, cat_val, cat_names)
    test_df = _build_design_matrix(X_test_raw, numeric, boolean, cat_test, cat_names)
    feature_names = list(train_df.columns)
    X_train = train_df.to_numpy(np.float32)
    X_val = val_df.to_numpy(np.float32)
    X_test = test_df.to_numpy(np.float32)
    fmin = X_train.min(axis=0).astype(np.float32)
    fmax = X_train.max(axis=0).astype(np.float32)
    frange = np.where(fmax > fmin, fmax - fmin, 1.0).astype(np.float32)

    return ProcessedDataset(
        name=active_config.name,
        X_train=X_train, X_validation=X_val, X_test=X_test,
        y_train=y_train.to_numpy(np.int64),
        y_validation=y_val.to_numpy(np.int64), y_test=y_test.to_numpy(np.int64),
        feature_names=feature_names, feature_min=fmin, feature_max=fmax, feature_range=frange,
        model_classes=np.asarray([0, 1], dtype=np.int64),
        raw_split={"train": X_train_raw, "validation": X_val_raw, "test": X_test_raw},
        metadata={
            "target_column": target,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "boolean_features": boolean,
            "onehot_feature_names": cat_names,
            # Informazioni indispensabili per ricreare il plot rosso/verde
            # del notebook con regole nuovamente leggibili.
            "onehot_mapping": _build_onehot_mapping(ohe, categorical) if categorical else {},
            "display_classes": (
                ["not_survived", "survived"]
                if dataset_name == "titanic"
                else ["benign", "malignant"]
                if dataset_name in {"breast", "breast_cancer"}
                else ["tested_negative", "tested_positive"]
                if config.name.lower() == "diabetes"
                else ["good", "bad"]
                if config.name.lower() in {"german", "german_credit", "credit_g"}
                else [_as_python_scalar(value) for value in classes]
            ),
        },
    )
