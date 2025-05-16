import { NextResponse } from 'next/server';

export async function POST(request) {
  const { message } = await request.json();

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: 'gpt-4',
      messages: [
        { role: 'system', content: 'Eres un asistente útil.' },
        { role: 'user', content: message }
      ]
    })
  });

  const data = await response.json();
  const reply = data.choices[0].message.content;

  return NextResponse.json({ reply });
}
