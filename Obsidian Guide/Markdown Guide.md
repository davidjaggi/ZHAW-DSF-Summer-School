# Markdown Guide

A extensible markdown reference guide.

## YAML Frontmatter

```yaml
---
title: "My Page"
date: 2024-01-01
tags: [markdown, guide]
---
```

## Table of Contents

- [Basic Syntax](#basic-syntax)
- [Text Formatting](#text-formatting)
- [Links and Images](#links-and-images)
- [Obsidian Links](#obsidian-links)
- [Lists](#lists)
- [Code](#code)
- [Tables](#tables)
- [Blockquotes](#blockquotes)
- [Horizontal Rules](#horizontal-rules)

---

## Basic Syntax

Heading levels and structure.

### Headers

```md
# H1 (Title)
## H2 (Section)
### H3 (Subsection)
#### H4 (Sub-subsection)
```

---

## Text Formatting

Bold, italic, and inline code.

### Bold

`**bold**` or `__bold__`

### Italic

`*italic*` or `_italic_`

### Inline Code

`` `inline code` ``

---

## Links and Images

Creating links and embedding images.

### Links

`[text](https://example.com)`

### Images

`![alt text](image.jpg)`

---

## Obsidian Links

- Internal link: `[[page]]`
- Link with alias: `[[page|alias]]`
- Link with hash: `[[page#tag]]`

---

## Lists

Ordered and unordered lists.

### Unordered

```md
- Item 1
- Item 2
  - Nested item
```

### Ordered

```md
1. First
2. Second
   1. Nested
3. Third
```

---

## Code

Code blocks and syntax highlighting.

### Inline Code

`` `code` ``

### Code Block

```python
def hello():
    print("Hello, World!")
```

```javascript
function greet(name) {
  return `Hello, ${name}!`;
}
```

---

## Tables

Creating tables with alignment.

```md
| Name | Age | City |
|------|-----|------|
| Alice | 30 | NY |
| Bob | 25 | LA |
```

---

## Blockquotes

> This is a blockquote.
> It can span multiple lines.

---

## Horizontal Rules

---

Three or more dashes create a horizontal rule.

---
