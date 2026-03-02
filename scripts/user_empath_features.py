from pathlib import Path
import pandas as pd
import numpy as np
from empath import Empath
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Optional: varimax rotation for interpretability
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(
            np.dot(Phi.T, np.asarray(Lambda)**3 - 
                   (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda)))))
        )
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)

def main():
    base_dir = Path(__file__).parent.parent
    in_csv = base_dir / "data" / "user_aggregate_conversations.csv"
    out_csv = base_dir / "data" / "user_with_empath_features.csv"
    out_factor_csv = base_dir / "data" / "user_with_empath_factors.csv"

    df = pd.read_csv(in_csv, on_bad_lines="warn")
    df["conversation"] = df["conversation"].fillna("").astype(str)

    lexicon = Empath()

    print("Extracting Empath categories per user...")
    empath_features = df["conversation"].apply(
        lambda x: lexicon.analyze(x, normalize=True)
    )

    empath_df = pd.DataFrame(list(empath_features))
    df_full = pd.concat([df, empath_df], axis=1)

    df_full.to_csv(out_csv, index=False)
    print("Saved full Empath feature file to:", out_csv)

    # ---- Optional: Factor analysis for dimensionality reduction ----
    # Remove non-feature columns
    feature_cols = empath_df.columns.tolist()
    X = empath_df.values

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_factors = 6  # adjust (4–8 typical)
    fa = FactorAnalysis(n_components=n_factors, random_state=0)
    Z = fa.fit_transform(Xs)

    # Rotate loadings for interpretability
    loadings = fa.components_.T
    rotated_loadings = varimax(loadings)

    # Attach factors
    for i in range(n_factors):
        df_full[f"factor_{i+1}"] = Z[:, i]

    df_full.to_csv(out_factor_csv, index=False)
    print("Saved Empath factor file to:", out_factor_csv)

    # Print top loading categories per factor
    print("\nTop loading categories per factor:")
    for i in range(n_factors):
        factor_loads = rotated_loadings[:, i]
        top_idx = np.argsort(np.abs(factor_loads))[-10:]
        top_categories = [feature_cols[j] for j in top_idx]
        print(f"Factor {i+1}: {top_categories}")

if __name__ == "__main__":
    main()