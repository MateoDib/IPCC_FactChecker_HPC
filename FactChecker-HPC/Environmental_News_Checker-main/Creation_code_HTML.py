import os
import json

def generate_html_from_json(json_dir, output_html, articles_data_dir):
    # Read all JSON files in the directory and load them into a dictionary
    articles_data = {}
    print(f"Reading JSON files from directory: {json_dir}")
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                # Use the filename without the extension as the article key
                article_key = os.path.splitext(filename)[0]
                articles_data[article_key] = article_data
            print(f"Loaded file: {filename} as key: {article_key}")
            print(f"Content of {filename}: {json.dumps(article_data, indent=4, ensure_ascii=False)}")

    # Directory to store individual JSON files for each article
    os.makedirs(articles_data_dir, exist_ok=True)
    print(f"Output directory for individual JSON files: {articles_data_dir}")

    # Save each article's data as a separate JSON file
    for article_key, article_data in articles_data.items():
        json_file_path = os.path.join(articles_data_dir, f"{article_key}.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=4, ensure_ascii=False)
        print(f"Created JSON file: {json_file_path}")
        print(f"Content of {json_file_path}: {json.dumps(article_data, indent=4, ensure_ascii=False)}")

    # Initialize HTML content
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Environmental News Checker</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            /* General Styling */
            body {{
                font-family: 'Roboto', sans-serif;
                background-color: #f9f9f9;
                margin: 0;
                padding: 0;
                color: #333;
            }}
            h1, h2 {{
                text-align: center;
                color: #2c3e50;
            }}
            .container {{
                width: 90%;
                max-width: 1000px;
                margin: 20px auto;
                padding: 20px;
                background: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
            }}
            select {{
                width: 100%;
                padding: 10px;
                margin: 20px 0;
                border-radius: 5px;
                border: 1px solid #ccc;
                font-size: 16px;
            }}
            .phrase {{
                background: #f4f4f4;
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.2s;
            }}
            .phrase:hover {{
                background: #e7f3ff;
                transform: scale(1.02);
            }}
            .score {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                margin-right: 5px;
                font-size: 0.9em;
            }}
            .score-high {{
                background-color: #d4edda;
                color: #155724;
            }}
            .score-medium {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .score-low {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .popup, .overlay {{
                display: none;
            }}
            .overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 10;
            }}
            .popup {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 8px 10px rgba(0, 0, 0, 0.2);
                z-index: 20;
                width: 80%;
                max-width: 500px;
            }}
            .close-btn {{
                float: right;
                cursor: pointer;
                font-size: 1.5em;
                color: #888;
            }}
            .close-btn:hover {{
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Environmental News Checker</h1>
            <p>Analyze environmental news articles for accuracy, bias, and more.</p>
            <select id="article-select" onchange="loadArticle(this.value)">
                <option value="">Select an article</option>"""

    # Add options for each article
    for article_key in articles_data.keys():
        html_content += f'<option value="{article_key}">{articles_data[article_key]["article_title"]}</option>\n'

    html_content += """</select>
            <div id="article-content"></div>
        </div>
        <div class="overlay" id="overlay" onclick="closePopup()"></div>
        <div class="popup" id="popup">
            <span class="close-btn" onclick="closePopup()">×</span>
            <div id="popup-content"></div>
        </div>
        <script>
            const articlesData = {};"""

    # Embed each article's JSON data
    for article_key, article_data in articles_data.items():
        html_content += f'articlesData["{article_key}"] = {json.dumps(article_data)};\n'

    html_content += """
            function loadArticle(articleKey) {
                const article = articlesData[articleKey];
                if (!article) return;

                const content = document.getElementById('article-content');
                content.innerHTML = '<h2>' + article.article_title + '</h2>';
                Object.keys(article.phrases).forEach(id => {
                    const phraseData = article.phrases[id];
                    const phraseElement = document.createElement('div');
                    phraseElement.classList.add('phrase');
                    phraseElement.innerHTML = phraseData.text;

                    // Add metrics
                    const metricsDiv = document.createElement('div');
                    metricsDiv.style.marginTop = "10px";
                    Object.keys(phraseData.analysis).forEach(metric => {
                        const metricData = phraseData.analysis[metric];
                        const scoreSpan = document.createElement('span');
                        scoreSpan.className = 'score ' + 
                            (metricData.score >= 4 ? 'score-high' : metricData.score >= 2 ? 'score-medium' : 'score-low');
                        scoreSpan.innerText = `${metric}: ${metricData.score !== null ? metricData.score : 'N/A'}`;
                        metricsDiv.appendChild(scoreSpan);
                    });
                    phraseElement.appendChild(metricsDiv);

                    phraseElement.onclick = () => showPopup(phraseData.analysis);
                    content.appendChild(phraseElement);
                });
            }

            function showPopup(analysis) {
                const popupContent = document.getElementById('popup-content');
                popupContent.innerHTML = '<h3>Phrase Analysis</h3>';
                Object.keys(analysis).forEach(metric => {
                    const metricData = analysis[metric];
                    popupContent.innerHTML += `
                        <div>
                            <strong>${metric}</strong>: ${metricData.score !== null ? metricData.score : 'N/A'}
                            <p>${metricData.justifications || 'No justification provided.'}</p>
                        </div>`;
                });

                document.getElementById('overlay').style.display = 'block';
                document.getElementById('popup').style.display = 'block';
            }

            function closePopup() {
                document.getElementById('overlay').style.display = 'none';
                document.getElementById('popup').style.display = 'none';
            }
        </script>
    </body>
    </html>"""

    # Write the HTML content to the output file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML file created at {output_html}")