# Viewing Documentation Locally

## Quick Start (Windows PowerShell)

### Step 1: Install Ruby (if not already installed)

Download and install Ruby from:
- **https://rubyinstaller.org/**
- Choose the latest Ruby 3.2+ version with DevKit

After installation, verify:
```powershell
ruby --version
```

### Step 2: Install Jekyll Dependencies

Open PowerShell in the project directory and run:

```powershell
cd docs
gem install bundler
bundle install
```

### Step 3: Run the Local Server

```powershell
bundle exec jekyll serve
```

You should see output like:
```
Configuration file: C:/Users/Public/embeddedtorch/docs/_config.yml
            Source: C:/Users/Public/embeddedtorch/docs
       Destination: C:/Users/Public/embeddedtorch/docs/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
                    done in 0.123 seconds.
 Auto-regeneration: enabled for 'C:/Users/Public/embeddedtorch/docs'
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl+c to stop.
```

### Step 4: Open in Browser

Go to: **http://localhost:4000** or **http://127.0.0.1:4000**

## Troubleshooting

### If `bundle install` fails:

Try:
```powershell
gem update bundler
bundle update
bundle install
```

### If port 4000 is already in use:

Use a different port:
```powershell
bundle exec jekyll serve --port 4001
```

Then access: **http://localhost:4001**

### If you get permission errors:

Run PowerShell as Administrator and try again.

### If Ruby is not recognized:

- Make sure Ruby is installed
- Check if Ruby is in your PATH environment variable
- Restart PowerShell after installation

## Live Reload

Jekyll automatically reloads when you make changes to files! Just:
1. Save your changes in any `.md` file
2. Refresh your browser
3. Changes appear immediately

## Stop the Server

Press `Ctrl + C` in the PowerShell window.

## Next Steps

- Edit any markdown file in `docs/` to update content
- Customize `_config.yml` for site settings
- View changes instantly in browser

