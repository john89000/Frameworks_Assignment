import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (cached to improve responsiveness on large files)
@st.cache_data
def load_data(path="metadata.csv"):
    df = pd.read_csv(path)
    # use get to avoid KeyError if column missing
    df["publish_time"] = pd.to_datetime(df.get("publish_time"), errors="coerce")
    df["year"] = df["publish_time"].dt.year
    return df

df = load_data()

st.title("CORD-19 Data Explorer")
st.write("An interactive exploration of COVID-19 research papers")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Year range slider
years = df["year"].dropna().astype(int)
if len(years) > 0:
    min_year, max_year = int(years.min()), int(years.max())
    # sensible default: full range
    default_start, default_end = min_year, max_year
    # if dataset contains recent years, keep a reasonable default view
    year_range = st.slider("Select year range", min_year, max_year, (default_start, default_end))

    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    if filtered.empty:
        st.info("No papers found in the selected year range. Try expanding the range or select 'Show raw data' to inspect the dataset.")
    else:

        # Publications per year
        st.subheader("Publications per Year")
        pubs_per_year = filtered["year"].value_counts().sort_index()
        # ensure all years in range are present (fill zeros)
        all_years = list(range(year_range[0], year_range[1] + 1))
        pubs_per_year = pubs_per_year.reindex(all_years, fill_value=0)
        st.line_chart(pubs_per_year)
        st.bar_chart(pubs_per_year)

        # Top journals
        st.subheader("Top Journals")
        if "journal" in filtered.columns:
            top_journals = filtered["journal"].fillna("(unknown)").value_counts().head(10)
            st.bar_chart(top_journals)
        else:
            st.info("No 'journal' column found in dataset.")

        # Word Cloud
        st.subheader("Word Cloud of Titles")
        titles = " ".join(filtered["title"].dropna().astype(str).tolist())
        if titles:
            try:
                from wordcloud import WordCloud

                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles)
                st.image(wordcloud.to_array())
            except Exception:
                st.warning("The 'wordcloud' package is not available — install it to see the word cloud (see requirements.txt).")
        else:
            st.info("No titles available for the selected range.")

        # Word Frequency Analysis
        st.subheader("Most Frequent Words in Titles")
        import collections
        import re
        if titles:
            words = re.findall(r"\w+", titles.lower())
            stopwords = set(["the", "and", "of", "in", "to", "a", "for", "on", "with", "by", "an", "at", "from", "as", "is", "are", "that", "this", "be", "or", "we", "using", "study", "covid", "19", "sars", "coronavirus"])
            filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
            word_counts = collections.Counter(filtered_words).most_common(10)
            word_freq_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
            st.table(word_freq_df)
        else:
            st.info("No title text available to compute word frequencies.")

        # Distribution of Paper Counts by Source
        # look for a column that contains 'source' if present
        source_col = next((c for c in filtered.columns if "source" in c.lower()), None)
        if source_col:
            st.subheader("Distribution of Paper Counts by Source")
            source_counts = filtered[source_col].fillna("(unknown)").value_counts().head(10)
            st.bar_chart(source_counts)
        else:
            st.info("No source-like column found to plot distribution by source.")
