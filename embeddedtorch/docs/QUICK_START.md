# Quick Start: Running Your Jekyll Documentation Site

## ⚡ TL;DR - Get Running in 3 Steps

### 1. Install Ruby (if needed)
- **Windows**: Download from https://rubyinstaller.org/
- **Mac**: Already installed, or use `brew install ruby`
- **Linux**: `sudo apt-get install ruby-full bundler`

### 2. Install Jekyll Dependencies
```bash
cd docs
gem install bundler
bundle install
```

### 3. Run the Site
```bash
bundle exec jekyll serve
```

Open **http://localhost:4000** in your browser! 🎉

---

## 📁 What's Included

Your documentation site includes:

- ✅ **Homepage** (`index.md`) - Main landing page
- ✅ **Getting Started** (`getting-started.md`) - Installation & basic usage
- ✅ **API Reference** (`api-reference.md`) - Complete API documentation
- ✅ **Layers** (`layers.md`) - All layer types explained
- ✅ **Operations** (`operations.md`) - Mathematical operations reference
- ✅ **Examples** (`examples.md`) - Working code examples
- ✅ **Jekyll Configuration** (`_config.yml`) - Site settings
- ✅ **GitHub Pages Setup** (`.github/workflows/jekyll.yml`) - Auto-deployment

## 🎨 Customization

### Edit Content
Just modify the `.md` files in the `docs/` directory. They use Markdown format with YAML front matter.

Example:
```yaml
---
layout: default
title: My New Page
---

# My Content Here
```

### Change Site Settings
Edit `docs/_config.yml`:
- Site title and description
- Navigation links
- Theme configuration

### Modify Theme
Edit `docs/assets/css/main.scss` for custom styling.

## 🚀 Deployment

### Automatic (GitHub Pages)
Your site automatically deploys when you push to the `main` branch:
```bash
git add .
git commit -m "Update docs"
git push
```

### Manual Build
```bash
cd docs
bundle exec jekyll build
# Output will be in docs/_site/
```

## 📖 Writing Documentation

### Adding a New Page
1. Create a new file: `docs/my-new-page.md`
2. Add front matter:
```yaml
---
layout: default
title: My New Page
---
```
3. Write content in Markdown
4. Add to navigation in `_config.yml`

### Code Examples
Use triple backticks:
````markdown
```python
from layers import EmbaeddableModel
model = EmbaeddableModel(torch.float32)
```
````

### Links
Link to other pages:
```markdown
[Getting Started]({{ "getting-started" | relative_url }})
```

## 🛠️ Common Issues & Solutions

### "Ruby not found"
- **Windows**: Run RubyInstaller
- **Mac/Linux**: Install with `brew` or `apt-get`

### "Bundle install fails"
```bash
gem update bundler
bundle update
```

### "Port 4000 in use"
```bash
bundle exec jekyll serve --port 4001
```

### "Could not locate Gemfile"
Make sure you're in the `docs/` directory:
```bash
cd docs
bundle exec jekyll serve
```

## 🎯 Next Steps

1. **Test Locally**: Run `bundle exec jekyll serve` and view at localhost:4000
2. **Customize**: Edit pages to match your project
3. **Deploy**: Push to GitHub to publish
4. **Iterate**: Update documentation as your project evolves

## 📚 More Help

- Full setup guide: See `JEKYLL_SETUP.md`
- Jekyll docs: https://jekyllrb.com/docs/
- Markdown guide: https://guides.github.com/features/mastering-markdown/

## 💡 Tips

- **Live Reload**: Jekyll auto-reloads when you save files
- **Drafts**: Create `docs/_drafts/` for unpublished posts
- **Assets**: Add images to `docs/assets/images/`
- **Syntax Highlighting**: Works automatically with fenced code blocks

---

Happy documenting! 🎉

