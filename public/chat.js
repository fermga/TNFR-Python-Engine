const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  // Mostrar el mensaje del usuario
  const userMessage = document.createElement('div');
  userMessage.textContent = `Tú: ${message}`;
  chatMessages.appendChild(userMessage);

  // Enviar el mensaje al backend
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message })
  });

  const data = await response.json();

  // Mostrar la respuesta del GPT
  const gptMessage = document.createElement('div');
  gptMessage.textContent = `GPT: ${data.reply}`;
  chatMessages.appendChild(gptMessage);

  userInput.value = '';
});
