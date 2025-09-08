import streamlit as st
import requests
from bs4 import BeautifulSoup
import langdetect
from langdetect import detect
import re
from urllib.parse import urlparse
import time
from typing import List, Dict, Tuple
import pandas as pd
from difflib import SequenceMatcher
import hashlib
from collections import defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="French Text Detector - Advanced",
    page_icon="🇫🇷",
    layout="wide"
)


def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def clean_text(text: str) -> str:
    """Clean text for better language detection"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-àáâãäçèéêëìíîïñòóôõöùúûüýÿ]', ' ', text)
    return text.strip()


def get_text_hash(text: str) -> str:
    """Generate hash for text deduplication"""
    return hashlib.md5(text.lower().encode()).hexdigest()


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def scrape_url(url: str, timeout: int = 10) -> Tuple[bool, str, List[str]]:
    """Scrape URL and return success status, error message, and text blocks"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text_blocks = []

        for element in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td']):
            text = element.get_text().strip()
            if text and len(text) > 20:
                cleaned_text = clean_text(text)
                if len(cleaned_text) > 20:
                    text_blocks.append(cleaned_text)

        return True, "", text_blocks

    except requests.exceptions.Timeout:
        return False, "Request timeout", []
    except requests.exceptions.ConnectionError:
        return False, "Connection error", []
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error: {e}", []
    except Exception as e:
        return False, f"Error: {str(e)}", []


def detect_french_blocks(text_blocks: List[str], url: str) -> List[Dict]:
    """Detect French text blocks and return them with metadata"""
    french_blocks = []

    for i, block in enumerate(text_blocks):
        try:
            if len(block.split()) < 5:
                continue

            detected_lang = detect(block)

            if detected_lang == 'fr':
                try:
                    lang_probs = langdetect.detect_langs(block)
                    fr_confidence = 0
                    for lang_prob in lang_probs:
                        if lang_prob.lang == 'fr':
                            fr_confidence = lang_prob.prob
                            break
                except:
                    fr_confidence = 0.5

                text_hash = get_text_hash(block)

                french_blocks.append({
                    'url': url,
                    'block_id': i + 1,
                    'text': block,
                    'confidence': fr_confidence,
                    'word_count': len(block.split()),
                    'char_count': len(block),
                    'hash': text_hash,
                    'text_preview': block[:100] + "..." if len(block) > 100 else block
                })

        except:
            continue

    return french_blocks


def deduplicate_blocks(all_blocks: List[Dict], similarity_threshold: float = 0.85) -> Dict:
    """Deduplicate blocks and group similar ones"""
    # Exact duplicates by hash
    hash_groups = defaultdict(list)
    for block in all_blocks:
        hash_groups[block['hash']].append(block)

    # Similar blocks (not exact duplicates)
    unique_blocks = []
    similarity_groups = []
    processed_hashes = set()

    for hash_key, blocks in hash_groups.items():
        if hash_key in processed_hashes:
            continue

        # Representative block (highest confidence)
        representative = max(blocks, key=lambda x: x['confidence'])
        group = {
            'representative': representative,
            'exact_duplicates': blocks,
            'similar_blocks': [],
            'total_occurrences': len(blocks),
            'urls': list(set([b['url'] for b in blocks]))
        }

        # Find similar blocks
        for other_hash, other_blocks in hash_groups.items():
            if other_hash == hash_key or other_hash in processed_hashes:
                continue

            other_rep = max(other_blocks, key=lambda x: x['confidence'])
            similarity = similarity_score(representative['text'], other_rep['text'])

            if similarity >= similarity_threshold:
                group['similar_blocks'].extend(other_blocks)
                group['total_occurrences'] += len(other_blocks)
                group['urls'].extend([b['url'] for b in other_blocks])
                processed_hashes.add(other_hash)

        group['urls'] = list(set(group['urls']))
        similarity_groups.append(group)
        processed_hashes.add(hash_key)

    return {
        'groups': similarity_groups,
        'total_unique_groups': len(similarity_groups),
        'total_blocks': len(all_blocks)
    }


def create_analytics_dashboard(dedup_result: Dict, all_blocks: List[Dict]):
    """Create analytics dashboard"""
    st.header("📊 Analytics Dashboard")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total French Blocks", dedup_result['total_blocks'])

    with col2:
        st.metric("Unique Content Groups", dedup_result['total_unique_groups'])

    duplicate_blocks = dedup_result['total_blocks'] - dedup_result['total_unique_groups']
    with col3:
        st.metric("Duplicate/Similar Blocks", duplicate_blocks)

    with col4:
        if dedup_result['total_blocks'] > 0:
            duplication_rate = (duplicate_blocks / dedup_result['total_blocks']) * 100
            st.metric("Duplication Rate", f"{duplication_rate:.1f}%")
        else:
            st.metric("Duplication Rate", "0%")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Top duplicated content
        groups_data = [(g['representative']['text_preview'], g['total_occurrences'])
                       for g in dedup_result['groups']]
        groups_data.sort(key=lambda x: x[1], reverse=True)

        if groups_data:
            top_groups = groups_data[:10]
            df_top = pd.DataFrame(top_groups, columns=['Content Preview', 'Occurrences'])

            fig = px.bar(df_top, x='Occurrences', y='Content Preview',
                         orientation='h', title="Top 10 Most Duplicated Content")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # URL distribution
        url_counts = Counter([block['url'] for block in all_blocks])
        df_urls = pd.DataFrame(list(url_counts.items()), columns=['URL', 'French Blocks Count'])
        df_urls = df_urls.sort_values('French Blocks Count', ascending=False).head(10)

        if not df_urls.empty:
            fig = px.bar(df_urls, x='French Blocks Count', y='URL',
                         orientation='h', title="Top 10 URLs by French Content")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🇫🇷 Advanced French Text Detector")
    st.markdown("**Scrape URLs, detect French text, and analyze content patterns across multiple pages**")

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'all_french_blocks' not in st.session_state:
        st.session_state.all_french_blocks = []
    if 'dedup_result' not in st.session_state:
        st.session_state.dedup_result = {}
    if 'failed_urls' not in st.session_state:
        st.session_state.failed_urls = []
    if 'processed_urls_count' not in st.session_state:
        st.session_state.processed_urls_count = 0

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    timeout = st.sidebar.slider("Request timeout (seconds)", 5, 30, 10)
    min_confidence = st.sidebar.slider("Minimum confidence for French detection", 0.1, 1.0, 0.7, 0.1)
    similarity_threshold = st.sidebar.slider("Similarity threshold for grouping", 0.5, 1.0, 0.85, 0.05)
    batch_size = st.sidebar.number_input("Batch processing size", 1, 50, 10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tips for large URL lists:**")
    st.sidebar.markdown("• Use smaller batch sizes for stability")
    st.sidebar.markdown("• Lower similarity threshold finds more groups")
    st.sidebar.markdown("• Results are automatically deduplicated")

    # Clear results button in sidebar
    if st.sidebar.button("🗑️ Clear Results"):
        st.session_state.analysis_complete = False
        st.session_state.all_french_blocks = []
        st.session_state.dedup_result = {}
        st.session_state.failed_urls = []
        st.session_state.processed_urls_count = 0
        st.rerun()

    # URL input
    st.header("📝 Enter URLs")

    urls_input = st.text_area(
        "Enter URLs (one per line):",
        height=150,
        placeholder="https://example1.com\nhttps://example2.com\n...",
        help="You can paste up to 300+ URLs. Processing will be done in batches for stability."
    )

    # Process URLs
    if st.button("🔍 Analyze URLs", type="primary"):
        if not urls_input.strip():
            st.error("Please enter at least one URL")
            return

        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        invalid_urls = [url for url in urls if not is_valid_url(url)]

        if invalid_urls:
            st.error(f"Invalid URLs detected: {', '.join(invalid_urls[:5])}{'...' if len(invalid_urls) > 5 else ''}")
            return

        st.info(f"Processing {len(urls)} URLs in batches of {batch_size}...")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_french_blocks = []
        failed_urls = []

        # Process in batches
        for batch_start in range(0, len(urls), batch_size):
            batch_end = min(batch_start + batch_size, len(urls))
            batch_urls = urls[batch_start:batch_end]

            for i, url in enumerate(batch_urls):
                current_index = batch_start + i
                status_text.text(
                    f"Processing batch {batch_start // batch_size + 1}/{(len(urls) - 1) // batch_size + 1}: {url}")

                success, error_msg, text_blocks = scrape_url(url, timeout)

                if success:
                    french_blocks = detect_french_blocks(text_blocks, url)
                    filtered_blocks = [block for block in french_blocks
                                       if block['confidence'] >= min_confidence]
                    all_french_blocks.extend(filtered_blocks)
                else:
                    failed_urls.append((url, error_msg))

                progress_bar.progress((current_index + 1) / len(urls))

            # Small delay between batches
            time.sleep(0.5)

        status_text.text("Analysis complete! Processing results...")

        # Store results in session state
        st.session_state.all_french_blocks = all_french_blocks
        st.session_state.failed_urls = failed_urls
        st.session_state.processed_urls_count = len(urls)

        # Deduplication and analysis
        if all_french_blocks:
            st.session_state.dedup_result = deduplicate_blocks(all_french_blocks, similarity_threshold)
            st.session_state.analysis_complete = True
        else:
            st.session_state.analysis_complete = True

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.all_french_blocks:
        # Only create analytics dashboard if dedup_result exists
        if st.session_state.dedup_result:
            create_analytics_dashboard(st.session_state.dedup_result, st.session_state.all_french_blocks)

        # Detailed Results
        st.header("🔍 Detailed Results")

        # View mode selection
        view_mode = st.radio(
            "Select view mode:",
            ["📋 Content Groups (Deduplicated)", "🌐 By URL", "📊 Data Table"],
            horizontal=True
        )

        if view_mode == "📋 Content Groups (Deduplicated)":
            st.subheader("Content Groups (Similar content grouped together)")

            # Check if dedup_result exists and has groups
            if st.session_state.dedup_result and 'groups' in st.session_state.dedup_result:
                sorted_groups = sorted(st.session_state.dedup_result['groups'],
                                       key=lambda x: x['total_occurrences'], reverse=True)

                for i, group in enumerate(sorted_groups):
                    rep = group['representative']

                    with st.expander(
                            f"Group #{i + 1} - {group['total_occurrences']} occurrences "
                            f"across {len(group['urls'])} URLs "
                            f"(Confidence: {rep['confidence']:.2f})"
                    ):
                        st.write("**Representative text:**")
                        st.write(rep['text'])

                        st.write(f"**Found on {len(group['urls'])} URLs:**")
                        for url in group['urls']:
                            st.write(f"• {url}")

                        if group['similar_blocks']:
                            st.write(f"**Includes {len(group['similar_blocks'])} similar variations**")
            else:
                st.error("Deduplication data not available. Please run the analysis again.")

        elif view_mode == "🌐 By URL":
            st.subheader("Results by URL")

            # Group blocks by URL
            url_groups = defaultdict(list)
            for block in st.session_state.all_french_blocks:
                url_groups[block['url']].append(block)

            for url, blocks in url_groups.items():
                with st.expander(f"{url} ({len(blocks)} French blocks)"):
                    for block in blocks:
                        st.write(f"**Block #{block['block_id']}** "
                                 f"(Confidence: {block['confidence']:.2f}, "
                                 f"Words: {block['word_count']})")
                        st.write(block['text'])
                        st.markdown("---")

        else:  # Data Table
            st.subheader("Complete Data Table")

            # Check if dedup_result exists and has groups
            if st.session_state.dedup_result and 'groups' in st.session_state.dedup_result:
                df_data = []
                for group in st.session_state.dedup_result['groups']:
                    rep = group['representative']
                    df_data.append({
                        'Content Preview': rep['text_preview'],
                        'Occurrences': group['total_occurrences'],
                        'URLs Count': len(group['urls']),
                        'Confidence': rep['confidence'],
                        'Word Count': rep['word_count'],
                        'Character Count': rep['char_count']
                    })

                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Deduplication data not available. Please run the analysis again.")

        # Export functionality
        st.header("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export deduplicated groups
            if st.session_state.dedup_result and 'groups' in st.session_state.dedup_result:
                export_groups_data = []
                for i, group in enumerate(st.session_state.dedup_result['groups']):
                    rep = group['representative']
                    export_groups_data.append({
                        'Group_ID': i + 1,
                        'Occurrences': group['total_occurrences'],
                        'URLs_Count': len(group['urls']),
                        'URLs': '; '.join(group['urls']),
                        'Confidence': rep['confidence'],
                        'Word_Count': rep['word_count'],
                        'Text': rep['text']
                    })

                if export_groups_data:
                    df_groups = pd.DataFrame(export_groups_data)
                    csv_groups = df_groups.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Deduplicated Groups (CSV)",
                        data=csv_groups,
                        file_name="french_content_groups.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Deduplicated groups export not available. Please run the analysis first.")

        with col2:
            # Export all blocks
            if st.session_state.all_french_blocks:
                df_all = pd.DataFrame(st.session_state.all_french_blocks)
                csv_all = df_all.to_csv(index=False)

                st.download_button(
                    label="📥 Download All Blocks (CSV)",
                    data=csv_all,
                    file_name="all_french_blocks.csv",
                    mime="text/csv"
                )

        # Failed URLs summary
        if st.session_state.failed_urls:
            st.header("⚠️ Failed URLs")
            st.write(f"{len(st.session_state.failed_urls)} URLs failed to process:")

            failed_df = pd.DataFrame(st.session_state.failed_urls, columns=['URL', 'Error'])
            st.dataframe(failed_df, use_container_width=True)

    elif st.session_state.analysis_complete and not st.session_state.all_french_blocks:
        st.warning("No French text blocks found in any of the processed URLs.")

        if st.session_state.failed_urls:
            st.error(f"{len(st.session_state.failed_urls)} URLs failed to process. Check your URLs and try again.")


if __name__ == "__main__":
    main()