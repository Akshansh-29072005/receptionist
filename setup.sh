#!/bin/bash
# SSIPMT Receptionist - ONE-CLICK SETUP
# Run: bash setup.sh

set -e  # Exit on error

echo "=================================================="
echo "🚀 SSIPMT RECEPTIONIST - AUTOMATIC SETUP"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check Python
echo "1️⃣  Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.8+"
    exit 1
fi
echo ""

# Step 2: Install Python dependencies
echo "2️⃣  Installing Python packages..."
$PYTHON_CMD -m pip install --upgrade pip > /dev/null 2>&1
$PYTHON_CMD -m pip install -q numpy opencv-python sounddevice flask requests faster-whisper
print_status "Core packages installed"
echo ""

# Step 3: Check/Install Ollama
echo "3️⃣  Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_status "Ollama already installed"
else
    print_warning "Ollama not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_warning "macOS detected. Please install Ollama manually:"
        echo "   brew install ollama"
        echo "   OR download from: https://ollama.com/download"
        read -p "Press Enter after installing Ollama..."
    else
        print_warning "Windows detected. Please install Ollama manually:"
        echo "   Download from: https://ollama.com/download"
        read -p "Press Enter after installing Ollama..."
    fi
fi
echo ""

# Step 4: Start Ollama
echo "4️⃣  Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    print_status "Ollama is already running"
else
    print_warning "Starting Ollama in background..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    if pgrep -x "ollama" > /dev/null; then
        print_status "Ollama started successfully"
    else
        print_error "Failed to start Ollama. Start manually: ollama serve"
    fi
fi
echo ""

# Step 5: Pull LLM model
echo "5️⃣  Downloading AI model (this may take a few minutes)..."
MODEL="llama3.2:1b"  # Smallest/fastest model
if ollama list | grep -q "$MODEL"; then
    print_status "Model '$MODEL' already downloaded"
else
    print_warning "Downloading $MODEL (~1.3GB)..."
    ollama pull $MODEL
    print_status "Model downloaded successfully"
fi
echo ""

# Step 6: Test Ollama
echo "6️⃣  Testing LLM connection..."
TEST_RESPONSE=$(curl -s http://localhost:11434/api/tags 2>&1)
if echo "$TEST_RESPONSE" | grep -q "models"; then
    print_status "LLM is ready!"
else
    print_warning "LLM may not be ready. Check with: ollama list"
fi
echo ""

# Step 7: Optional - Install Piper TTS
echo "7️⃣  Checking Piper TTS..."
if command -v piper &> /dev/null; then
    print_status "Piper TTS already installed"
else
    print_warning "Piper TTS not found (needed for voice output)"
    echo "   Install from: https://github.com/rhasspy/piper"
    echo "   Or skip if you don't need voice output"
fi
echo ""

# Final instructions
echo "=================================================="
echo "✅ SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "📋 QUICK START:"
echo "   1. Run: $PYTHON_CMD ssipmt_final.py"
echo "   2. Open browser: http://127.0.0.1:8000"
echo "   3. Say 'Hello Robot' to start"
echo ""
echo "🔧 CONFIGURATION:"
echo "   • Change LLM model in ssipmt_final.py (line 38)"
echo "   • Available models: llama3.2:1b, llama3.2, phi3, mistral"
echo "   • Install larger model: ollama pull llama3.2"
echo ""
echo "📁 REQUIRED DIRECTORIES:"
echo "   • ./models/       - AI models"
echo "   • ./voices/       - TTS voices (amy, pratham)"
echo "   • ./assets/       - Videos and logo"
echo "   • ./known_faces/  - Face recognition (optional)"
echo ""
echo "🆘 TROUBLESHOOTING:"
echo "   • Ollama not connecting: ollama serve"
echo "   • Model not found: ollama pull llama3.2:1b"
echo "   • Test LLM: ollama run llama3.2:1b 'hello'"
echo ""
print_status "Ready to run! Execute: $PYTHON_CMD ssipmt_final.py"
echo "=================================================="