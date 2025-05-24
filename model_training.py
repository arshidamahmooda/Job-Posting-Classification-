from sklearn.cluster import AgglomerativeClustering

def preprocess_and_cluster(df, n_clusters=5):
    df['Skills'] = df['Skills'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(X.toarray())  # AgglomerativeClustering needs dense array

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    df.to_csv("clustered_jobs.csv", index=False)
    return df
