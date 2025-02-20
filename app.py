import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def execute_code(code_str):
    try:
        # Create a list to store outputs
        outputs = []
        
        # Create a custom print function that captures output
        def custom_print(*args, **kwargs):
            outputs.append(' '.join(map(str, args)))
        
        # Set up execution environment with our custom print
        exec_globals = {
            'torch': torch,
            'nn': nn,
            'np': np,
            'plt': plt,
            'print': custom_print,
            'outputs': outputs
        }
        
        # Execute the code
        exec(code_str, exec_globals)
        
        # Format and return all captured outputs
        if outputs:
            return "\n".join(outputs)
        else:
            return "Code executed but no output was generated."
            
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n💡 Tip: Check your syntax and make sure all required modules are imported."

def main():
    st.title("Intro to Pytorch for Machine Learning")
    
    with st.expander("📚 Introduction"):
        st.markdown("""
   # Welcome to Intro to PyTorch for Machine Learning

In this lab, you'll embark on a journey into the world of deep learning using PyTorch, a powerful framework that enables rapid experimentation and efficient model development. Our mission is to guide you through the foundational concepts and practical skills necessary to build an AI assistant—one that can process and understand complex data like text, audio, and images, much like the human brain.

## Overview

Neural networks are inspired by the biological neural networks of the human brain. They consist of layers of interconnected neurons, each with its own weights and biases, that work together to process input data and generate meaningful outputs. Throughout this lab, you will explore key aspects of neural network design, including:

- **Neurons and Layers:** Discover the basic building blocks of neural networks. Learn how individual neurons process inputs through weights, biases, and activation functions.
- **Activation Functions:** Understand why non-linear activation functions such as ReLU, Sigmoid, Tanh, and Softmax are essential for capturing complex patterns in data.
- **Feedforward Neural Networks:** Construct and train a simple feedforward network that transforms raw inputs into actionable outputs—similar to how an AI assistant interprets user commands.
- **Model Parameters and Weights:** Delve into the inner workings of model parameters. Learn how adjustments to weights and biases directly influence the network's predictions and overall performance.
- **Data Handling with Tensors:** Master the use of tensors, PyTorch's core data structure, which facilitates efficient computation and automatic differentiation.
- **Training a Neural Network:** Gain practical experience in training a model. Learn about loss functions, gradient descent, and optimization techniques that iteratively improve the network's performance.
- **Evaluating and Visualizing Performance:** Measure your model's accuracy and visualize training progress through loss and accuracy curves, helping you diagnose issues like overfitting or underfitting.

## What You Will Achieve

By the end of this lab, you will:
- Develop a strong conceptual foundation of how neural networks operate.
- Gain hands-on experience implementing various components of neural networks in PyTorch.
- Learn how to adjust and optimize model parameters to enhance performance.
- Understand the complete cycle of training, validating, and evaluating a model—skills essential for building robust AI assistants.

Whether you're aspiring to create a smarter virtual assistant or simply eager to deepen your understanding of deep learning, this comprehensive introduction sets the stage for your journey into building the AI of the future.

Let's dive in and start transforming raw data into intelligent insights!
        """)
    
    with st.expander("🔢 Step 1: Understanding Neurons and Layers"):
        st.write("### 🧑‍🏫 What is a Neuron?")
        st.write("Imagine a neuron as a decision-making unit. Just like in the brain, it takes inputs, processes them using weights and biases, and decides whether to activate based on an activation function.")
        st.write("Each neuron in an AI assistant's brain processes different types of data—words, images, or numbers—to make intelligent decisions.")
        
        st.write("### 🔍 Understanding Weights and Bias")
        st.write("A neuron has two key components that determine how it processes information:")
        st.markdown("""
        - **Weight (w):** Determines the importance of an input. Higher weights mean the input has a stronger effect on the output.
        - **Bias (b):** A value that shifts the neuron's output. It allows the neuron to be activated even if the input is small or zero.
        
        Think of it this way:
        - The **weight** is like a volume knob that adjusts the importance of an input.
        - The **bias** is like a baseline level of activation—ensuring that even weak signals can be processed.
        
        If you were designing an AI assistant that responds to voice commands, the weight would amplify important words, while the bias ensures the assistant still responds even if the words are spoken softly.
        """)
        
        st.write("### 🔍 Understanding Neuron Output")
        st.write("The output of a neuron determines how strongly it activates:")
        st.markdown("""
        - **Low Output (Close to 0):** The neuron is mostly inactive, meaning the input did not strongly trigger a response. For an AI assistant, this could mean the spoken command was too weak to recognize.
        - **High Output:** The neuron is fully activated, meaning the input strongly influenced the result. In an AI assistant, this could indicate a clear, loud voice command that the system recognizes easily.
        - **Medium Output:** The neuron is partially activated, meaning the input had some effect but was not dominant.
        
        This behavior is crucial in deep learning, where multiple neurons interact to refine decision-making.
        """)
        
        st.write("### 🔬 Adjust Weights and Bias")
        st.write("In this simulation, you can control the neuron's weight and bias to see how they affect the output. Think of this as tuning the way your AI assistant processes data.")
        x = st.slider("Input (x)", -5.0, 5.0, 1.0)
        w = st.slider("Weight (w)", -3.0, 3.0, 2.0)
        b = st.slider("Bias (b)", -3.0, 3.0, 0.5)
        
        def neuron(x, w, b):
            # Ensure inputs are tensors
            x = torch.tensor([x]) if isinstance(x, (int, float)) else x
            w = torch.tensor([w]) if isinstance(w, (int, float)) else w
            b = torch.tensor([b]) if isinstance(b, (int, float)) else b
            
            output = torch.relu(torch.matmul(x, w) + b)
            print(f"A neuron transforms input {x.item()} using weight {w.item()} and bias {b.item()} to produce output {output.item()}")
            return output
        
        output = neuron(x, w, b)
        st.write(f"Neuron Output: {output}")
        
        st.write("###  🛠️ Implement a Neuron in Python")
        st.write("Now, let's implement this concept in code. This is how your AI assistant processes raw input using a simple neuron.")
        code = """
import torch

def neuron(x, w, b):
    # Ensure inputs are tensors
    x = torch.tensor([x]) if isinstance(x, (int, float)) else x
    w = torch.tensor([w]) if isinstance(w, (int, float)) else w
    b = torch.tensor([b]) if isinstance(b, (int, float)) else b
    
    output = torch.relu(torch.matmul(x, w) + b)
    print(f"A neuron transforms input {x.item()} using weight {w.item()} and bias {b.item()} to produce output {output.item()}")
    return output

# Test the neuron
x = 1.0  # Input signal
w = 2.0  # Weight
b = 0.5  # Bias
output = neuron(x, w, b)
"""
        st.code(code, language="python")
        if st.button("Run Code", key="neuron_code"):
            st.write(execute_code(code))
        
        st.write("### 🧪 Try This!")
        st.write("Try different combinations of inputs, weights, and biases to understand how neurons behave:")

        code_examples = """
import torch

def neuron(x, w, b):
    # Ensure inputs are tensors
    x = torch.tensor([x]) if isinstance(x, (int, float)) else x
    w = torch.tensor([w]) if isinstance(w, (int, float)) else w
    b = torch.tensor([b]) if isinstance(b, (int, float)) else b
    
    output = torch.relu(torch.matmul(x, w) + b)
    print(f"A neuron transforms input {x.item()} using weight {w.item()} and bias {b.item()} to produce output {output.item()}")
    return output

# Example 1: Positive weight and bias
print("\\nExample 1: Standard positive values")
x, w, b = 1.0, 2.0, 0.5
output = neuron(x, w, b)

# Example 2: Negative weight
print("\\nExample 2: Negative weight")
x, w, b = 1.0, -2.0, 0.5
output = neuron(x, w, b)

# Example 3: Large bias
print("\\nExample 3: Large positive bias")
x, w, b = 1.0, 2.0, 5.0
output = neuron(x, w, b)

# Example 4: Large negative weight
print("\\nExample 4: Large negative weight")
x, w, b = 1.0, -5.0, 0.5
output = neuron(x, w, b)
"""

        st.code(code_examples, language="python")
        if st.button("Run Examples", key="neuron_examples"):
            st.write(execute_code(code_examples))

        st.markdown("""
### 💡 Key Observations:

1. **Positive Weights (Example 1)**
   - Amplifies the input signal
   - Output increases with larger weights
   - Common in features that positively contribute to the decision

2. **Negative Weights (Example 2)**
   - Reduces or blocks the input signal
   - ReLU prevents negative outputs (sets them to zero)
   - Useful for inhibitory connections

3. **Large Bias (Example 3)**
   - Creates a baseline activation
   - Can make neuron "fire" even with small inputs
   - Helps control activation threshold

4. **Large Negative Weight (Example 4)**
   - Strongly inhibits the input
   - Often results in zero output due to ReLU
   - Shows how neurons can "turn off" certain inputs
""")
        
        st.write("### ❓ Comprehension Question")
        st.write("What happens if the bias `b` is very large compared to `w * x`?")
        options = [
            "The neuron remains inactive (output is zero).",
            "The output is always positive.",
            "The neuron behaves randomly.",
            "The neuron stops learning."
        ]
        answer = st.selectbox("Select the correct answer:", options)
        if st.button("Check Answer", key="check_answer_neuron"):
            if answer == "The output is always positive.":
                st.success("Correct! A large bias can push the neuron into always being activated.")
            else:
                st.error("Not quite. Try again!")
                st.write("💡 Hint: Think about how ReLU handles large inputs. If bias is very large, ReLU ensures that the output remains positive.")
    
    with st.expander("🔢 Step 2: Exploring Activation Functions"):
        st.write("### 🧑‍🏫 Why Do We Need Activation Functions?")
        st.write("Activation functions allow neural networks to learn complex patterns by introducing non-linearity.")
        
        st.write("### 🔍 Understanding Activation Functions in Context")
        st.markdown("""
        - **ReLU (Rectified Linear Unit):** Think of ReLU as a decision-maker for an AI assistant. If a word in a sentence is relevant, ReLU ensures it is passed through without modification. If it is not relevant, ReLU blocks it out by setting it to zero.
        - **Sigmoid:** Useful for making yes/no decisions, such as whether a detected sound is speech or noise.
        - **Tanh:** Strengthens the impact of certain features by pushing values toward -1 or 1, often improving speech emotion detection.
        - **Softmax:** Converts multiple outputs into probabilities, such as determining the most likely user intent from different possible commands.
        """)
        
        st.write("### 🔬 Visualizing Activation Functions")
        x = torch.linspace(-5, 5, 100)
        activation_functions = {
            'ReLU': torch.relu(x),
            'Sigmoid': torch.sigmoid(x),
            'Tanh': torch.tanh(x),
            'Softmax': torch.nn.functional.softmax(x, dim=0)
        }
        
        plt.figure(figsize=(8, 6))
        for name, y in activation_functions.items():
            plt.plot(x.numpy(), y.numpy(), label=name)
        plt.legend()
        plt.title("Activation Functions")
        plt.xlabel("Input")
        plt.ylabel("Output")
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting
        
        st.write("### 🛠️ Implement Activation Functions in a Neural Network Layer")
        st.write("Now, let's integrate activation functions into a neural network layer to see how they transform data.")
        code = """
import torch.nn as nn

class ActivationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        return x

model = ActivationNN()
output = model(torch.tensor([[1.0, 2.0]]))
"""
        st.code(code, language="python")
        if st.button("Run Code", key="activation_nn_code"):
            st.write(execute_code(code))
        
        st.write("###  🛠️ Compare Activation Functions")
        st.write("Let's explore how different activation functions transform the same input data. Below are examples using various activation functions:")

        code_examples = """
import torch
import torch.nn as nn

# Input data
x = torch.tensor([[1.0, 2.0]])

# Test different activation functions
class ActivationComparison(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3)
        
        # Define different activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Apply linear transformation
        x = self.layer(x)
        
        # Apply different activations and store results
        results = {
            'Linear (no activation)': x,
            'ReLU': self.relu(x),
            'Sigmoid': self.sigmoid(x),
            'Tanh': self.tanh(x),
            'Softmax': self.softmax(x)
        }
        return results

# Create model and get outputs
model = ActivationComparison()
outputs = model(x)

# Print results
for name, result in outputs.items():
    print(f"{name}:\\n{result}\\n")
"""

        st.code(code_examples, language="python")
        if st.button("Run Activation Comparison", key="activation_comparison"):
            st.write(execute_code(code_examples))

        st.markdown("""
### 🧪 Try This!
1. Experiment with different inputs:
   ```python
   # Test negative inputs
   input_tensor = torch.tensor([[-1.0, -2.0]])
   
   # Test extreme values
   input_tensor = torch.tensor([[-10.0, 10.0]])
   
   # Test mixed values
   input_tensor = torch.tensor([[-1.5, 2.5]])
   
   # Test near-zero values
   input_tensor = torch.tensor([[-0.1, 0.1]])
   ```

### 💡 Key Observations:
1. **ReLU Behavior**
   - Zeros out negative values
   - Preserves positive values unchanged
   - Creates sparse activations

2. **Sigmoid Properties**
   - Squashes all values between 0 and 1
   - Smooth transition around zero
   - Saturates for extreme values

3. **Tanh Characteristics**
   - Squashes values between -1 and 1
   - Steeper gradient than sigmoid
   - Zero-centered output

4. **Layer Size Effects**
   - More neurons capture more patterns
   - Wider layers allow more feature combinations
   - Deeper networks learn hierarchical features

### 🔍 What to Watch For:
- **Confidence Levels**: Higher numbers indicate stronger preferences
- **Response Balance**: How the AI distributes attention between:
  - Providing information
  - Asking questions
  - Taking actions
- **Decision Making**: How quickly and confidently the AI chooses a response

### 💡 Tips:
- Start with ReLU for hidden layers
- Use sigmoid for binary classification outputs
- Use tanh when zero-centered outputs are important
- Monitor for dead neurons (always output zero)
- Combine different activations based on your needs
""")
        
        st.write("### ❓ Comprehension Question")
        options = [
            "ReLU outputs negative values unchanged.",
            "Sigmoid maps values between 0 and 1.",
            "Tanh has a range of 0 to 1.",
            "Softmax is best for binary classification."
        ]
        answer = st.selectbox("Which statement about activation functions is correct?", options)
        if st.button("Check Answer", key="activation_question"):
            if answer == "Sigmoid maps values between 0 and 1.":
                st.success("Correct! The Sigmoid function squashes values into a range between 0 and 1.")
            else:
                st.error("Not quite. Try again!")
                st.write("💡 Hint: Think about how Sigmoid is used in binary classification models.")
    
    with st.expander("🔢 Step 3: Building a Feedforward Neural Network"):
        st.write("### 🧑‍🏫 What is a Feedforward Neural Network?")
        st.write("A feedforward neural network is the backbone of many AI systems, including AI assistants. In these systems, data flows from the input layer to the output layer through one or more hidden layers without looping back. This structure enables the network to process raw inputs—such as user queries or voice commands—and transform them into meaningful outputs.")
        
        st.write("###  🛠️ Building a Feedforward Neural Network")
        st.write("Let's create a neural network that processes data through multiple layers and see how the data transforms at each step:")

        code = """
import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x):
        # Process through layers
        out = self.fc1(x)
        out = self.relu(out)
        output = self.fc2(out)
        
        # Explain the output in AI assistant context
        print("\\n• Understanding the Output in AI Assistant Context:")
        print("-"*40)
        
        response_types = [
            "Providing Information",
            "Asking Clarifying Question",
            "Taking Action"
        ]
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(output, dim=1)
        
        # Show confidence levels with visual bars
        print("\\nConfidence Levels:")
        for i, (prob, response_type) in enumerate(zip(probs[0], response_types)):
            percentage = prob.item() * 100
            bars = "█" * int(percentage / 5)  # Visual representation
            print(f"• {response_type}: {percentage:.1f}% {bars}")
        
        # Show decision
        max_prob_idx = torch.argmax(probs).item()
        print(f"\\nBased on these scores, the AI would: {response_types[max_prob_idx]}")
        
        return output

# Network configuration
input_features = 4
hidden_neurons = 5
output_classes = 3

# Create and run model
model = FeedforwardNN(input_features, hidden_neurons, output_classes)
sample_input = torch.tensor([[1.0, 0.5, -0.2, 0.8]])
output = model(sample_input)
"""

        st.code(code, language="python")
        if st.button("Run Feedforward NN Code", key="feedforward_nn_code"):
            st.write(execute_code(code))

        st.markdown("""
        ### 🧪 Try This!

        Here's how to adjust the model's behavior by changing its configuration:

        #### 1. Create an Inquisitive AI
        ```python
        # Increase hidden neurons to recognize uncertainty
        model = FeedforwardNN(input_size=4, hidden_size=8, output_size=3)
        ```
        This configuration makes the AI more likely to ask clarifying questions when uncertain. Great for:
        - Complex user requests
        - Ambiguous commands
        - Ensuring accurate understanding

        #### 2. Create a Decisive AI
        ```python
        # Reduce hidden neurons for simpler, direct responses
        model = FeedforwardNN(input_size=4, hidden_size=3, output_size=3)
        ```
        This makes the AI more direct and action-oriented. Better for:
        - Simple commands
        - Clear instructions
        - Quick responses

        #### 3. Create an Informative AI
        ```python
        # Bias input towards information provision
        sample_input = torch.tensor([[1.5, 0.2, -0.3, 0.1]])
        ```
        This makes the AI prioritize providing information. Useful for:
        - Educational contexts
        - Detailed explanations
        - Teaching scenarios

        #### 4. Create an Action-Oriented AI
        ```python
        # Bias input towards taking action
        sample_input = torch.tensor([[0.1, -0.2, 1.5, 0.3]])
        ```
        This makes the AI focus on executing commands. Good for:
        - Task execution
        - Command processing
        - Getting things done

        #### 5. Create a Balanced AI
        ```python
        # Use moderate values for balanced behavior
        sample_input = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        ```
        This creates an AI that balances all response types. Ideal for:
        - General-purpose use
        - Mixed interactions
        - Versatile applications

        ### 🔍 What to Watch For:
        - **Confidence Levels**: Higher numbers indicate stronger preferences
        - **Response Balance**: How the AI distributes attention between:
          - Providing information
          - Asking questions
          - Taking actions
        - **Decision Making**: How quickly and confidently the AI chooses a response

        ### 💡 Tips:
        - Start with the balanced configuration
        - Adjust gradually to find the right behavior
        - Monitor how changes affect the AI's responses
        - Find the balance that works best for your needs
        """)

        st.markdown("""
        ### ❓ Comprehension Question
        
        What would be the best network configuration for an AI assistant that needs to handle complex user requests requiring detailed follow-up questions?
        """)
        
        options = [
            "A network with fewer hidden neurons (hidden_size=3) for direct responses",
            "A network with more hidden neurons (hidden_size=8) for nuanced understanding",
            "A balanced network (hidden_size=5) with action-oriented input",
            "A network with equal input values but no hidden layer"
        ]
        
        answer = st.selectbox("Select the best configuration:", options)
        
        if st.button("Check Answer", key="check_answer_feedforward"):
            if answer == "A network with more hidden neurons (hidden_size=8) for nuanced understanding":
                st.success("✅ Correct! More hidden neurons give the network greater capacity to recognize uncertainty and ask appropriate follow-up questions. This is ideal for handling complex requests that require clarification.")
                st.markdown("""
                **Explanation:**
                - More hidden neurons = More pattern recognition capacity
                - Better at detecting ambiguity in user requests
                - More likely to ask clarifying questions when needed
                - Ideal for complex interactions requiring detailed understanding
                """)
            else:
                st.error("❌ Not quite. Think about how the number of hidden neurons affects the network's ability to recognize complex patterns and uncertainty.")
                st.write("💡 Hint: Consider which configuration would be better at detecting subtle nuances in user requests that might need clarification.")
    
    with st.expander("🔢 Step 4: Understanding Model Parameters and Weights"):
        st.write("### 🧑‍🏫 The Role of Parameters in AI Assistants")
        st.write("Just as humans develop different personalities and response styles, an AI assistant's behavior is shaped by its parameters - weights and biases. Weights determine how strongly the AI responds to different inputs (like paying more attention to urgent requests), while biases represent the AI's default tendencies (like being naturally helpful even with minimal input).")
        
        st.write("### 🛠️  Inspecting AI Assistant Parameters")
        code = """
import torch
import torch.nn as nn

class AIAssistant(nn.Module):
    def __init__(self):
        super(AIAssistant, self).__init__()
        # Simple model to demonstrate personality traits
        self.personality = nn.Linear(3, 3)  # 3 inputs: urgency, clarity, complexity
        
    def forward(self, x):
        response = self.personality(x)
        
        # Convert to response probabilities
        probs = torch.nn.functional.softmax(response, dim=1)
        
        # Explain the assistant's behavior
        traits = ['Helpful', 'Analytical', 'Action-oriented']
        print("AI Assistant Personality Analysis:")
        print("-" * 40)
        for trait, prob in zip(traits, probs[0]):
            percentage = prob.item() * 100
            bars = "█" * int(percentage / 5)
            print(f"{trait:15} : {percentage:4.1f}% {bars}")
        
        return response

# Create and analyze AI assistant
assistant = AIAssistant()

# Test with different input scenarios
print("\\nScenario 1: Urgent, Clear Request")
urgent_input = torch.tensor([[0.9, 0.8, 0.3]])
output = assistant(urgent_input)

print("\\nScenario 2: Complex, Unclear Request")
complex_input = torch.tensor([[0.3, 0.2, 0.9]])
output = assistant(complex_input)
"""

        st.code(code, language="python")
        if st.button("Run AI Assistant Analysis", key="params_code"):
            st.write(execute_code(code))

        st.markdown("""
        ### 🧪 Try This!

        Here's how different parameter configurations shape your AI's behavior:

        #### 1. Responsive Assistant (High Weights)
        ```python
        # Configure assistant to be highly responsive to input
        with torch.no_grad():
            assistant.personality.weight.fill_(2.0)  # Strong response to input
            assistant.personality.bias.fill_(0.0)    # Neutral baseline
        ```
        Effects:
        - Very responsive to user needs
        - Strong reactions to urgency
        - Clear differentiation between requests
        - Good for: Priority-based response systems

        #### 2. Stable Assistant (Low Weights)
        ```python
        # Configure assistant to be more measured
        with torch.no_grad():
            assistant.personality.weight.fill_(0.5)  # Dampened responses
            assistant.personality.bias.fill_(0.0)    # Neutral baseline
        ```
        Effects:
        - More consistent responses
        - Less affected by input variations
        - Stable behavior across situations
        - Good for: General-purpose assistance

        #### 3. Proactive Assistant (Positive Bias)
        ```python
        # Configure assistant to be naturally helpful
        with torch.no_grad():
            assistant.personality.weight.fill_(1.0)    # Normal responsiveness
            assistant.personality.bias.fill_(0.5)      # Helpful baseline
        ```
        Effects:
        - Naturally inclined to help
        - Offers suggestions proactively
        - Higher baseline engagement
        - Good for: Customer service, educational settings

        #### 4. Analytical Assistant (Selective Weights)
        ```python
        # Configure assistant to focus on complex queries
        with torch.no_grad():
            assistant.personality.weight[1] *= 1.5     # Emphasis on analysis
            assistant.personality.bias.fill_(-0.2)     # Slight skepticism
        ```
        Effects:
        - Thorough analysis of requests
        - More likely to ask clarifying questions
        - Careful consideration before action
        - Good for: Technical support, research assistance

        #### 5. Balanced Assistant
        ```python
        # Configure assistant for balanced behavior
        with torch.no_grad():
            assistant.personality.weight.fill_(1.0)    # Standard responsiveness
            assistant.personality.bias.fill_(0.0)      # Neutral baseline
        ```
        Effects:
        - Balanced response to all situations
        - Equal consideration of options
        - Adaptable to different contexts
        - Good for: General-purpose AI assistance

        ### 🔍 What to Watch For:
        - **Response Patterns**: How the assistant reacts to different inputs
        - **Consistency**: Stability of responses across situations
        - **Proactiveness**: Tendency to initiate interactions
        - **Adaptability**: Balance between different response types

        ### 💡 Tips for AI Assistant Development:
        - Match parameters to intended use case
        - Test with diverse input scenarios
        - Monitor for unintended biases
        - Adjust based on user feedback
        """)

        st.markdown("""
        ### 💡 Key Observations:
        1. **Weight Behavior**
           - Larger weights create stronger connections
           - Small weights make gradual changes
           - Weight signs determine excitation/inhibition

        2. **Bias Impact**
           - Positive bias creates default activation
           - Negative bias raises activation threshold
           - Zero bias relies purely on weighted inputs

        3. **Parameter Scaling**
           - Too large: may cause instability
           - Too small: may slow learning
           - Balanced: promotes stable learning

        4. **Initialization Effects**
           - Random initialization breaks symmetry
           - Proper scaling prevents vanishing/exploding gradients
           - Initial values influence final solution
        """)

        st.markdown("""
        ### ❓ Comprehension Question
        
        If you want to create an AI assistant that proactively offers help even before users ask detailed questions, which parameter adjustment would be most effective?
        """)
        
        options = [
            "Increase weights to amplify all inputs",
            "Add a positive bias to create a helpful baseline tendency",
            "Decrease weights to make responses more consistent",
            "Set all parameters to their default values"
        ]
        
        answer = st.selectbox("Select the best approach:", options)
        
        if st.button("Check Answer", key="check_answer_params"):
            if answer == "Add a positive bias to create a helpful baseline tendency":
                st.success("✅ Correct! Adding a positive bias creates a default tendency to be helpful, making the AI assistant naturally proactive even with minimal user input.")
                st.markdown("""
                **Explanation:**
                - Positive bias creates a helpful baseline personality
                - Assistant will engage even with minimal prompting
                - Maintains responsiveness while being proactive
                - Ideal for creating an engaging, helpful AI assistant
                """)
            else:
                st.error("❌ Not quite. Think about which parameter would make the AI naturally inclined to help, even without strong input signals.")
                st.write("💡 Hint: Consider what makes a helpful person offer assistance before being asked.")
    
    with st.expander("🔢 Step 5: Data Handling with Tensors"):
        st.write("### 🧑‍🏫 What are Tensors in AI Assistants?")
        st.write("Tensors are how your AI assistant processes and understands information. Just like a human brain processes different types of sensory input, tensors help the AI handle various types of data—from text commands to voice inputs to image recognition. They're the building blocks that enable your AI to make sense of user interactions.")
        
        st.write("### 🛠️ Working with Tensors in an AI Assistant")
        code = """
import torch

# Simulate different types of AI assistant inputs
print("1. Processing Different Input Types:\\n")

# Text command processing
text_command = torch.tensor([[0.2, 0.8, 0.1],   # "Help me with..."
                           [0.9, 0.2, 0.7]])     # "I need to..."
print("Text command features:\\n", text_command)

# Voice input features
voice_pattern = torch.rand(2, 4)  # Pitch, volume, speed, clarity
print("\\nVoice input features:\\n", voice_pattern)

# User context information
user_context = torch.tensor([1.0, 0.0, 0.5])  # Time, urgency, complexity
print("\\nUser context:\\n", user_context)

print("\\n2. AI Response Processing:\\n")

# Generate response probabilities
responses = torch.tensor([[0.8, 0.1, 0.1],    # Informative response
                         [0.2, 0.7, 0.1],    # Clarifying question
                         [0.1, 0.2, 0.7]])   # Action response

print("Response probabilities:\\n", responses)
print("\\nMost likely response type:", torch.argmax(responses, dim=1))

# Combine different features for final decision
combined = torch.cat([text_command[0], user_context])
print("\\nCombined features for decision:\\n", combined)
"""

        st.code(code, language="python")
        if st.button("Run AI Assistant Tensor Code", key="tensor_code"):
            st.write(execute_code(code))

        st.markdown("""
        ### 🧪 Try This!

        Here's how tensors handle different aspects of AI assistant interactions:

        #### 1. Processing User Input
        ```python
        # Process text command
        command = torch.tensor([0.9, 0.2, 0.7])  # Command features
        
        # Process voice characteristics
        voice = torch.tensor([0.8, 0.6, 0.4, 0.7])  # Voice features
        
        # Combine multimodal input
        combined_input = torch.cat([command, voice])
        ```
        Applications:
        - Text command understanding
        - Voice pattern recognition
        - Multimodal processing

        #### 2. Context Handling
        ```python
        # User context tensor
        context = torch.tensor([
            0.8,  # Time of day
            0.3,  # User's recent activity
            0.9   # Conversation history
        ])
        
        # Combine with current input
        enriched_input = torch.cat([input_tensor, context])
        ```
        Benefits:
        - Contextual awareness
        - Personalized responses
        - Better understanding of user intent

        #### 3. Response Generation
        ```python
        # Response type probabilities
        responses = torch.tensor([
            0.7,  # Provide information
            0.2,  # Ask question
            0.1   # Take action
        ])
        
        # Apply softmax for decision
        decision = torch.nn.functional.softmax(responses, dim=0)
        ```
        Features:
        - Response type selection
        - Confidence scoring
        - Decision making

        #### 4. Memory Management
        ```python
        # Store conversation history
        history = torch.stack([
            previous_interaction,
            current_interaction
        ])
        
        # Update context
        updated_context = torch.mean(history, dim=0)
        ```
        Capabilities:
        - Conversation tracking
        - Context updating
        - Memory retention

        ### 🔍 What to Watch For:
        - **Input Processing**: How different input types are encoded
        - **Context Integration**: How context affects responses
        - **Response Selection**: How decisions are made
        - **Memory Updates**: How information is retained

        ### 💡 Tips for AI Assistant Development:
        - Normalize input features for consistent processing
        - Combine multiple input modalities effectively
        - Maintain appropriate context window size
        - Balance response probabilities
        """)

        st.markdown("""
        ### ❓ Comprehension Question
        
        When processing both text and voice input in an AI assistant, which tensor operation would be most appropriate for combining these different input modalities?
        """)
        
        options = [
            "torch.add() to sum the features",
            "torch.multiply() to amplify common features",
            "torch.cat() to preserve all feature information",
            "torch.mean() to average the features"
        ]
        
        answer = st.selectbox("Select the best operation:", options)
        
        if st.button("Check Answer", key="check_answer_tensors"):
            if answer == "torch.cat() to preserve all feature information":
                st.success("✅ Correct! Concatenation (torch.cat()) preserves all the unique features from both text and voice inputs, allowing the AI assistant to consider all available information.")
                st.markdown("""
                **Explanation:**
                - Concatenation keeps all features intact
                - Allows the model to learn from both modalities
                - Preserves the distinct characteristics of each input type
                - Essential for multimodal AI processing
                """)
            else:
                st.error("❌ Not quite. Think about which operation would maintain all the important features from both input types.")
                st.write("💡 Hint: Consider how a human processes both written and spoken information - we don't combine or average them, we consider all aspects separately.")
    
    with st.expander("🔢 Step 6: Training a Neural Network"):
        st.write("### 🧑‍🏫 Training an AI Assistant")
        st.write("Training is how your AI assistant learns to understand and respond to users effectively. Just like a human learning through practice and feedback, the AI uses training data to improve its responses, learning from each interaction to become more helpful and accurate.")
        
        st.write("### 🛠️ Training Process Demonstration")
        code = """
import torch
import torch.nn as nn
import torch.optim as optim

class AssistantTrainer:
    def __init__(self):
        # Simple assistant model
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # 3 response types
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def train_step(self, input_data, target):
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(input_data)
        loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), output

# Create trainer
trainer = AssistantTrainer()

# Simulate training data
# Input features: [urgency, clarity, complexity, user_history]
inputs = torch.tensor([
    [0.9, 0.8, 0.3, 0.7],  # Urgent, clear request
    [0.3, 0.4, 0.9, 0.5],  # Complex, unclear request
    [0.6, 0.7, 0.4, 0.8]   # Moderate request
], dtype=torch.float32)

# Target responses: 0=inform, 1=clarify, 2=act
targets = torch.tensor([2, 1, 0])  # Expected responses

print("Training AI Assistant...")
print("-" * 40)

# Training loop
for epoch in range(5):
    total_loss = 0
    for i in range(len(inputs)):
        loss, output = trainer.train_step(inputs[i], targets[i])
        total_loss += loss
        
        # Show training progress
        probs = torch.nn.functional.softmax(output, dim=0)
        response_types = ['Inform', 'Clarify', 'Act']
        
        print(f"\\nEpoch {epoch+1}, Sample {i+1}")
        print(f"Input: Urgency={inputs[i][0]:.1f}, Clarity={inputs[i][1]:.1f}, "
              f"Complexity={inputs[i][2]:.1f}, History={inputs[i][3]:.1f}")
        print("Response Probabilities:")
        for j, (resp_type, prob) in enumerate(zip(response_types, probs)):
            bars = "█" * int(prob.item() * 20)
            print(f"{resp_type:7}: {prob.item():.3f} {bars}")
    
    print(f"\\nEpoch {epoch+1} Average Loss: {total_loss/len(inputs):.4f}")
"""

        st.code(code, language="python")
        if st.button("Run Training Demo", key="training_code"):
            st.write(execute_code(code))

        st.markdown("""
        ### 🧪 Try This!

        Here's how to modify the training process for different assistant behaviors:

        #### 1. Customer Service Assistant
        ```python
        # Configure for customer service
        training_data = {
            'inputs': torch.tensor([
                [0.9, 0.8, 0.2, 0.7],  # Clear customer question
                [0.5, 0.3, 0.8, 0.6],  # Complex inquiry
                [0.8, 0.7, 0.3, 0.9]   # Follow-up question
            ]),
            'targets': torch.tensor([0, 1, 2])  # inform, clarify, act
        }
        ```
        Focus:
        - Quick response to clear questions
        - Careful handling of complex issues
        - Proactive follow-up

        #### 2. Technical Support Assistant
        ```python
        # Configure for technical support
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([
            0.5,  # Lower weight for simple information
            1.5,  # Higher weight for clarification
            1.0   # Normal weight for actions
        ]))
        ```
        Emphasis:
        - Detailed problem understanding
        - Thorough troubleshooting
        - Step-by-step guidance

        #### 3. Educational Assistant
        ```python
        # Configure for educational purposes
        learning_rate = 0.001  # Slower, more careful learning
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        ```
        Benefits:
        - Thorough explanation of concepts
        - Patient response style
        - Progressive complexity handling

        #### 4. Task-Oriented Assistant
        ```python
        # Configure for task execution
        model = nn.Sequential(
            nn.Linear(4, 12),  # Larger hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),   # Add regularization
            nn.Linear(12, 3)
        )
        ```
        Features:
        - Efficient task processing
        - Clear action steps
        - Focused responses

        ### 🔍 What to Watch For During Training:
        - **Loss Trends**: How quickly the assistant improves
        - **Response Distribution**: Balance between different actions
        - **Overfitting Signs**: Too specific to training examples
        - **Generalization**: Performance on new situations

        ### 💡 Tips for Training:
        - Start with diverse training examples
        - Monitor response quality regularly
        - Adjust learning rate based on progress
        - Use validation data to check generalization
        """)

        st.markdown("""
        ### ❓ Comprehension Question
        
        When training an AI assistant for customer service, which training configuration would be most appropriate?
        """)
        
        options = [
            "Fast learning rate with minimal examples",
            "Balanced dataset with emphasis on clarification responses",
            "Technical-focused dataset with complex problems only",
            "Simple dataset with only direct answers"
        ]
        
        answer = st.selectbox("Select the best training approach:", options)
        
        if st.button("Check Answer", key="check_answer_training"):
            if answer == "Balanced dataset with emphasis on clarification responses":
                st.success("✅ Correct! A balanced dataset with emphasis on clarification ensures the assistant can handle various situations while maintaining good customer communication.")
                st.markdown("""
                **Explanation:**
                - Balanced data covers diverse customer scenarios
                - Emphasis on clarification improves understanding
                - Better customer satisfaction through accurate responses
                - Reduces need for multiple interactions
                """)
            else:
                st.error("❌ Not quite. Think about what makes a good customer service representative effective.")
                st.write("💡 Hint: Consider how important it is to properly understand customer needs before providing solutions.")
    
    with st.expander("🔢 Step 7: Evaluating and Visualizing Performance"):
        st.write("### 🧑‍🏫 Evaluating and Visualizing AI Assistant Performance")
        st.write("Just like monitoring a human assistant's learning progress, evaluating your AI assistant is crucial to ensure it's learning effectively. By tracking performance metrics and visualizing them, you can identify if your assistant is improving, struggling, or needs adjustments to better serve users.")
        
        st.write("### 🔍 Understanding: Performance Metrics and Visualization")
        st.markdown("""
        - **Validation Dataset:** Real user interactions used to test the assistant's performance
        - **Loss Curve:** Shows how well the assistant is learning from conversations
        - **Accuracy Curve:** Tracks how often the assistant provides appropriate responses
        
        These visualizations help ensure your AI assistant is continuously improving and maintaining high-quality interactions.
        """)
        
        st.write("### 🛠️ Evaluating and Visualizing Performance")
        st.write("Below is a visualization of how an AI assistant's performance improves over training epochs:")
        code = """
import matplotlib.pyplot as plt
import numpy as np

# Simulated AI assistant metrics over 10 epochs
epochs = list(range(1, 11))

# Different learning scenarios
scenarios = {
    'Ideal Learning': {
        'training_loss': [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.24, 0.22],
        'validation_loss': [0.95, 0.8, 0.6, 0.5, 0.45, 0.42, 0.4, 0.38, 0.37, 0.36],
        'accuracy': [60, 65, 70, 72, 75, 78, 80, 82, 83, 85]
    }
}

def analyze_learning(scenario_data):
    # Calculate key metrics
    final_train_loss = scenario_data['training_loss'][-1]
    final_val_loss = scenario_data['validation_loss'][-1]
    final_accuracy = scenario_data['accuracy'][-1]
    
    # Calculate improvements
    loss_improvement = (scenario_data['training_loss'][0] - final_train_loss) / scenario_data['training_loss'][0] * 100
    accuracy_improvement = scenario_data['accuracy'][-1] - scenario_data['accuracy'][0]
    
    # Calculate learning stability
    loss_stability = np.std(scenario_data['training_loss'])
    
    print("AI Assistant Learning Analysis:")
    print("-" * 40)
    print(f"Final Performance Metrics:")
    print(f"- Training Loss: {final_train_loss:.3f}")
    print(f"- Validation Loss: {final_val_loss:.3f}")
    print(f"- Response Accuracy: {final_accuracy:.1f}%")
    print(f"\\nLearning Progress:")
    print(f"- Loss Reduction: {loss_improvement:.1f}%")
    print(f"- Accuracy Improvement: +{accuracy_improvement:.1f}%")
    print(f"- Learning Stability: {loss_stability:.3f} (lower is better)")
    
    # Evaluate learning quality
    gen_gap = final_val_loss - final_train_loss
    print(f"\\nGeneralization Analysis:")
    print(f"- Generalization Gap: {gen_gap:.3f}")
    
    if gen_gap > 0.2:
        print("⚠️ Warning: Potential overfitting detected")
    elif gen_gap < 0.05:
        print("✅ Good generalization: Model learns effectively")
    
    # Learning recommendations
    print("\\nRecommendations:")
    if loss_stability > 0.2:
        print("- Consider reducing learning rate for more stable training")
    if accuracy_improvement < 20:
        print("- May need more training epochs for better performance")
    if gen_gap > 0.1:
        print("- Consider adding regularization to prevent overfitting")

def plot_learning_curves(scenario_data):
    plt.figure(figsize=(12, 5))
    
    # Plotting learning progress
    plt.subplot(1, 2, 1)
    plt.plot(epochs, scenario_data['training_loss'], 
             label='Training Loss', marker='o')
    plt.plot(epochs, scenario_data['validation_loss'], 
             label='Validation Loss', marker='o')
    plt.xlabel('Conversation Rounds')
    plt.ylabel('Response Quality Loss')
    plt.title('AI Assistant Learning Progress')
    plt.legend()
    
    # Plotting response accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, scenario_data['accuracy'], 
             label='Response Accuracy', color='green', marker='o')
    plt.xlabel('Conversation Rounds')
    plt.ylabel('Accurate Responses (%)')
    plt.title('AI Assistant Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot and analyze ideal learning scenario
print("Ideal Learning Scenario:")
plot_learning_curves(scenarios['Ideal Learning'])
analyze_learning(scenarios['Ideal Learning'])
"""

        st.code(code, language="python")
        if st.button("Run Evaluation Code", key="evaluation_code"):
            st.write(execute_code(code))

        st.markdown("""
        ### 🧪 Try This!

        #### 1. Ideal Learning Pattern
        ```python
        # Configure for steady, effective learning
        metrics = {
            'training_loss': [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.24, 0.22],
            'validation_loss': [0.95, 0.8, 0.6, 0.5, 0.45, 0.42, 0.4, 0.38, 0.37, 0.36],
            'accuracy': [60, 65, 70, 72, 75, 78, 80, 82, 83, 85]
        }
        ```
        Characteristics:
        - Steady decrease in both losses
        - Consistent accuracy improvement
        - Small gap between training and validation
        - Good for: General-purpose AI assistants

        #### 2. Overfitting Pattern
        ```python
        # Pattern showing memorization
        metrics = {
            'training_loss': [0.9, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005],
            'validation_loss': [0.95, 0.6, 0.5, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0],
            'accuracy': [60, 75, 85, 90, 92, 93, 93, 93, 93, 93]
        }
        ```
        Warning Signs:
        - Training loss continues to decrease
        - Validation loss starts increasing
        - Accuracy plateaus
        - Bad for: Real-world deployment

        #### 3. Underfitting Pattern
        ```python
        # Pattern showing insufficient learning
        metrics = {
            'training_loss': [0.9, 0.85, 0.82, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74],
            'validation_loss': [0.95, 0.9, 0.87, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8, 0.79],
            'accuracy': [55, 57, 58, 59, 60, 60, 61, 61, 62, 62]
        }
        ```
        Warning Signs:
        - Both losses decrease very slowly
        - Low accuracy with minimal improvement
        - Similar performance on training and validation
        - Indicates: Need for model capacity increase

        #### 4. Unstable Learning Pattern
        ```python
        # Pattern showing learning instability
        metrics = {
            'training_loss': [0.9, 0.4, 0.7, 0.3, 0.6, 0.2, 0.5, 0.3, 0.4, 0.3],
            'validation_loss': [0.95, 0.5, 0.8, 0.4, 0.7, 0.3, 0.6, 0.4, 0.5, 0.4],
            'accuracy': [60, 75, 65, 80, 70, 85, 75, 80, 77, 81]
        }
        ```
        Warning Signs:
        - Fluctuating losses and accuracy
        - Inconsistent improvement
        - Unpredictable performance
        - Indicates: Need for learning rate adjustment

        ### 🔍 What to Watch For:
        - **Loss Convergence**: Both losses should steadily decrease
        - **Generalization Gap**: Gap between training and validation loss
        - **Accuracy Trends**: Steady improvement without plateaus
        - **Learning Stability**: Consistent, predictable improvements

        ### 💡 Tips for Evaluation:
        - Monitor multiple metrics simultaneously
        - Look for early signs of overfitting
        - Check for learning stability
        - Validate on diverse conversation samples
        """)

        st.markdown("""
        ### ❓ Comprehension Question
        
        What is the main reason for monitoring both training and validation loss during AI assistant development?
        """)
        
        options = [
            "To make the assistant respond faster",
            "To detect if the assistant is truly learning or just memorizing responses",
            "To increase the number of possible responses",
            "To reduce the training time needed"
        ]
        
        answer = st.selectbox("Select the correct answer:", options)
        
        if st.button("Check Answer", key="check_answer_evaluation"):
            if answer == "To detect if the assistant is truly learning or just memorizing responses":
                st.success("✅ Correct! Monitoring both losses helps ensure your AI assistant is learning to generalize well to new conversations, rather than just memorizing training examples.")
                st.markdown("""
                **Why this matters:**
                - Helps create more adaptable AI assistants
                - Ensures better responses to new questions
                - Improves overall user experience
                - Guides training improvements
                """)
            else:
                st.error("❌ Not quite. Think about what makes an AI assistant truly helpful in real conversations.")
                st.write("💡 Hint: Consider the difference between memorizing exact responses and understanding how to handle new situations.")

    with st.expander("📝 Conclusion: Your AI Assistant Journey"):
        st.write("### 🎓 What We've Learned")
        st.markdown("""
        Throughout this tutorial, we've covered essential aspects of building AI assistants:

        #### 1. Neural Network Foundations
        - Understanding neural network architecture
        - How layers process information
        - Activation functions and their roles
        - Building blocks of AI assistants

        #### 2. Parameters and Weights
        - Role of weights in decision-making
        - Impact of biases on default behaviors
        - How parameters shape assistant personality
        - Fine-tuning response patterns

        #### 3. Data Handling with Tensors
        - Processing different input types
        - Managing conversation context
        - Combining multiple data sources
        - Efficient data representation

        #### 4. Training Process
        - Learning from user interactions
        - Adapting to different scenarios
        - Balancing various response types
        - Improving through feedback

        #### 5. Performance Evaluation
        - Monitoring learning progress
        - Measuring response quality
        - Analyzing user satisfaction
        - Identifying areas for improvement
        """)

    with st.expander("⭐ Lab Feedback"):
            st.write("### Help Us Improve!")
            st.write("How would you rate this lab?")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                rating = st.slider(
                    "Rate from 1 to 5 stars:",
                    min_value=1,
                    max_value=5,
                    value=5,
                    help="1 = Needs Improvement, 5 = Excellent"
                )
                
                # Display stars based on rating
                stars = "⭐" * rating
                st.write(f"Your rating: {stars}")
            
            with col2:
                # Show different messages based on rating
                if rating == 5:
                    st.success("Thank you! We're glad you found the lab excellent!")
                elif rating == 4:
                    st.success("Thanks! We appreciate your positive feedback!")
                elif rating == 3:
                    st.info("Thank you for your feedback. We'll keep improving!")
                else:
                    st.warning("We appreciate your honest feedback. We'll work on making it better!")
            
            # Optional feedback text
            feedback_text = st.text_area(
                "Additional feedback (optional):",
                height=100,
                placeholder="Share any specific comments, suggestions, or topics you'd like to see covered..."
            )
            
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback! Your input helps us improve the learning experience.")
                # Here you would typically save the feedback to a database
                # For now, we'll just display a thank you message


if __name__ == "__main__":
    main()
