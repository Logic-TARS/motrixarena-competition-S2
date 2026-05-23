from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "微信公众平台.html"
CLEAN = ROOT / "clean.html"
TEXT = ROOT / "article.txt"


class ContentExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.in_content = False
        self.depth = 0
        self.parts = []
        self.text_parts = []

    def _attrs(self, attrs):
        return dict(attrs)

    def _render_attrs(self, attrs):
        pairs = []
        attr_map = self._attrs(attrs)
        if "data-src" in attr_map:
            attr_map["src"] = attr_map["data-src"]
            attr_map.pop("data-src", None)
        for name, value in attr_map.items():
            if value is None:
                pairs.append(escape(name))
            else:
                pairs.append(f'{escape(name)}="{escape(value, quote=True)}"')
        return (" " + " ".join(pairs)) if pairs else ""

    def handle_starttag(self, tag, attrs):
        attr_map = self._attrs(attrs)
        if not self.in_content and tag == "div" and attr_map.get("id") == "js_content":
            self.in_content = True
            self.depth = 1
            self.parts.append(f"<{tag}{self._render_attrs(attrs)}>")
            return

        if self.in_content:
            self.parts.append(f"<{tag}{self._render_attrs(attrs)}>")
            if tag not in {"br", "img", "input", "meta", "link", "hr", "source"}:
                self.depth += 1

    def handle_startendtag(self, tag, attrs):
        if self.in_content:
            self.parts.append(f"<{tag}{self._render_attrs(attrs)} />")

    def handle_endtag(self, tag):
        if not self.in_content:
            return
        self.parts.append(f"</{tag}>")
        self.depth -= 1
        if self.depth == 0:
            self.in_content = False

    def handle_data(self, data):
        if self.in_content:
            self.parts.append(escape(data))
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def handle_entityref(self, name):
        if self.in_content:
            self.parts.append(f"&{name};")
            self.text_parts.append(unescape(f"&{name};"))

    def handle_charref(self, name):
        if self.in_content:
            self.parts.append(f"&#{name};")
            self.text_parts.append(unescape(f"&#{name};"))


def find(pattern, html, default=""):
    match = re.search(pattern, html, re.S)
    return unescape(match.group(1)).strip() if match else default


def main():
    html = SOURCE.read_text(encoding="utf-8")
    parser = ContentExtractor()
    parser.feed(html)

    title = find(r'<title>(.*?)</title>', html, "微信文章")
    author = find(r'id="js_author_name_text">(.*?)</span>', html)
    publish_time = find(r'id="publish_time"[^>]*>(.*?)</em>', html)
    content = "".join(parser.parts)
    content = re.sub(r'class="([^"]*)js_img_placeholder([^"]*)"', r'class="\1\2"', content)

    clean_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f5f7fb;
      color: #141414;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
    }}
    main {{
      width: min(760px, calc(100% - 28px));
      margin: 28px auto;
      background: #fff;
      padding: 28px 0 44px;
      box-shadow: 0 10px 40px rgba(15, 30, 60, .08);
    }}
    .meta {{
      padding: 0 28px 24px;
      border-bottom: 1px solid #edf0f5;
      margin-bottom: 24px;
    }}
    .meta h1 {{
      margin: 0 0 12px;
      font-size: 28px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .meta p {{
      margin: 0;
      color: #687385;
      font-size: 14px;
    }}
    #js_content img {{
      max-width: 100% !important;
      height: auto !important;
    }}
    #js_content {{
      overflow-wrap: anywhere;
    }}
    @media (max-width: 640px) {{
      main {{
        width: 100%;
        margin: 0;
        box-shadow: none;
      }}
      .meta {{
        padding-left: 18px;
        padding-right: 18px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="meta">
      <h1>{escape(title)}</h1>
      <p>{escape(author)} {escape(publish_time)}</p>
    </header>
    {content}
  </main>
</body>
</html>
"""

    CLEAN.write_text(clean_html, encoding="utf-8")
    lines = [title, f"{author} {publish_time}".strip(), ""]
    lines.extend(re.sub(r"\s+", " ", item).strip() for item in parser.text_parts)
    TEXT.write_text("\n".join(line for line in lines if line), encoding="utf-8")


if __name__ == "__main__":
    main()
