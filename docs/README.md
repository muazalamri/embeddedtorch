# EmbeddedTorch Documentation

This directory contains the documentation site for EmbeddedTorch, built with Jekyll and Markdown.

## Building the Site Locally

### Prerequisites

- Ruby 3.2+
- Bundler gem

### Setup

1. Install dependencies:
```bash
cd docs
bundle install
```

2. Build and serve locally:
```bash
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000`

### Building for Production

```bash
bundle exec jekyll build
```

## Documentation Structure

- `index.md` - Homepage
- `getting-started.md` - Quick start guide
- `api-reference.md` - API documentation
- `layers.md` - Layer reference
- `operations.md` - Operations reference
- `examples.md` - Working examples

## Deployment

This site is automatically deployed to GitHub Pages via GitHub Actions. The configuration is in `.github/workflows/jekyll.yml`.

### Manual Deployment

If you need to deploy manually:

1. Build the site:
```bash
cd docs
bundle exec jekyll build
```

2. The built site will be in `docs/_site/`

## Contributing

To add or modify documentation:

1. Edit the appropriate `.md` file in the `docs/` directory
2. Test locally with `bundle exec jekyll serve`
3. Commit and push changes
4. GitHub Actions will automatically rebuild and deploy

