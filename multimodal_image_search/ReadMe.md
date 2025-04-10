# Multimodal Image Search Application

## What is this project?
This is a web application that allows you to search through images using both text descriptions and visual content. Think of it like a smart image search engine that understands both what you type and what the images look like.

## Features
- 🔍 Search images using text descriptions
- 🖼️ Search images using visual similarity
- 🌐 Web-based interface that's easy to use
- 🚀 Fast and efficient search capabilities

## Prerequisites
Before you can run this application, you'll need:
1. **Docker Desktop** installed on your computer
   - Download it from [Docker's official website](https://www.docker.com/products/docker-desktop)
   - Follow the installation instructions for Windows

## Installation Steps
1. **Start Docker Desktop**
   - Open Docker Desktop from your Start menu
   - Wait until it shows "Docker Desktop is running" in the system tray

2. **Open Command Prompt or PowerShell**
   - Press Windows + R
   - Type "cmd" or "powershell" and press Enter

3. **Navigate to the project folder**
   - Use the `cd` command to go to where you saved this project
   - For example: `cd D:\Data Science\Code`

4. **Build the application**
   - Run this command:
   ```
   docker build --pull --rm -f 'multimodal_docker.dockerfile' -t 'multimodal_image_search' '.'
   ```

5. **Start the application**
   - Run this command:
   ```
   docker-compose up
   ```

## Using the Application
1. Once the application is running, open your web browser
2. Go to: `http://localhost:8501`
3. You'll see a user-friendly interface where you can:
   - Upload images to the search database
   - Search for images using text descriptions
   - Search for images using visual similarity

## Troubleshooting
If you encounter any issues:

1. **Docker not starting**
   - Make sure Docker Desktop is running
   - Try restarting Docker Desktop
   - Check if virtualization is enabled in your BIOS

2. **Build errors**
   - Make sure all files are in the correct location
   - Check that Docker Desktop is running properly
   - Try running the build command again

3. **Application not accessible**
   - Make sure port 8501 is not being used by another application
   - Check that the application started successfully
   - Try refreshing your browser

## Support
If you need help:
- Check the troubleshooting section above
- Make sure you have the latest version of Docker Desktop
- Contact the development team for assistance

## Technical Details (For Developers)
This application uses:
- Python 3.9
- Streamlit for the web interface
- Transformers for text and image processing
- ChromaDB for vector storage
- Docker for containerization

## License
MIT License

Copyright (c) 2025 Hari Krishna Kunderu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
