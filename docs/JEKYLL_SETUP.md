# Jekyll Setup Guide for EmbeddedTorch Documentation

This guide will help you set up and run the Jekyll documentation site locally.

## Quick Start

### Windows Setup

1. **Install Ruby**
   - Download RubyInstaller from https://rubyinstaller.org/
   - Install Ruby 3.2+ and the MSYS2 development toolchain
   - Verify installation:
     ```powershell
     ruby --version
     ```

2. **Install Jekyll and dependencies**
   ```powershell
   cd docs
   gem install bundler
   bundle install
   ```

3. **Run the site**
   ```powershell
   bundle exec jekyll serve
   ```

4. **Open your browser**
   - Navigate to http://localhost:4000

### Linux/Mac Setup

1. **Install Ruby** (if not already installed)
   ```bash
   # Linux
   sudo apt-get install ruby-full build-essential zlib1g-dev
   
   # Mac
   brew install ruby
   ```

2. **Install dependencies**
   ```bash
   cd docs
   gem install bundler
   bundle install
   ```

3. **Run the site**
   ```bash
   bundle exec jekyll serve
   ```

4. **Access the site**
   - Open http://localhost:4000 in your browser

## Troubleshooting

### "Could not locate Gemfile" Error

Make sure you're in the `docs/` directory:
```bash
cd docs
bundle install
```

### Permission Errors (Linux/Mac)

If you get permission errors, try:
```bash
sudo gem install bundler
```

Or install gems to your user directory:
```bash
gem install bundler --user-install
export PATH="$HOME/.gem/bin:$PATH"
```

### Bundle Install Fails

Try updating bundler:
```bash
gem update bundler
bundle update
```

### Port Already in Use

If port 4000 is in use, specify a different port:
```bash
bundle exec jekyll serve --port 4001
```

## Editing Documentation

### Modify Existing Pages

Edit the markdown files in the `docs/` directory:
- `index.md` - Homepage
- `getting-started.md` - Getting started guide
- `api-reference.md` - API documentation
- `layers.md` - Layer reference
- `operations.md` - Operations reference
- `examples.md` - Example code

### Add New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add the front matter:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   ---
   ```
3. Add the page to `_config.yml` navigation section
4. View at `http://localhost:4000/your-page-name`

### Adding Blog Posts

Create files in `docs/_posts/` with format:
```
YYYY-MM-DD-title.md
```

Example: `2024-01-15-new-feature.md`

## Site Configuration

Edit `docs/_config.yml` to customize:
- Site title and description
- Navigation links
- Theme settings
- GitHub Pages URL

## Building for Production

Generate the static site:
```bash
bundle exec jekyll build
```

Output will be in `docs/_site/`

## GitHub Pages Deployment

The site is automatically deployed via GitHub Actions when you:
1. Push to the `main` branch
2. The workflow (`.github/workflows/jekyll.yml`) builds and deploys the site

### Manual Deployment

If you need to deploy manually:
```bash
bundle exec jekyll build
# Upload _site folder to your web server
```

## Common Commands

```bash
# Install dependencies
bundle install

# Run local server
bundle exec jekyll serve

# Run with drafts
bundle exec jekyll serve --drafts

# Build production site
bundle exec jekyll build

# Clean and rebuild
bundle exec jekyll clean
bundle exec jekyll build

# Check configuration
bundle exec jekyll doctor
```

## Directory Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── _layouts/            # HTML layouts
├── _includes/           # Reusable components
├── _posts/              # Blog posts
├── index.md            # Homepage
├── getting-started.md   # Getting started guide
├── api-reference.md     # API docs
├── layers.md           # Layer reference
├── operations.md       # Operations reference
├── examples.md         # Examples
├── Gemfile             # Ruby dependencies
└── README.md           # This guide

```

## Markdown Tips

### Code Blocks

\`\`\`python
# Python code
from layers import EmbaeddableModel
\`\`\`

### Links

```markdown
[Getting Started]({{ "getting-started" | relative_url }})
[External Link](https://example.com)
```

### Images

```markdown
![Alt text]({{ "/assets/images/example.png" | relative_url }})
```

### Tables

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

## Next Steps

1. Customize the theme and colors in `_config.yml`
2. Add your own content to the documentation pages
3. Test locally with `bundle exec jekyll serve`
4. Push changes to GitHub to deploy

## Need Help?

- Jekyll docs: https://jekyllrb.com/docs/
- Minima theme: https://github.com/jekyll/minima
- GitHub Pages: https://pages.github.com/

