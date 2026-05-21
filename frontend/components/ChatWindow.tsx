import React, { useState } from 'react';

interface Message {
  role: 'user' | 'bot';
  text: string;
}

export default function ChatWindow() {
  const [applicationId, setApplicationId] = useState<string>('APP-001');
  const [message, setMessage] = useState<string>('');
  const [history, setHistory] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage: Message = { role: 'user', text: message };
    setHistory((prev) => [...prev, userMessage]);
    setMessage('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          application_id: applicationId,
          message: message,
        }),
      });

      if (!response.ok) throw new Error('Network payload rejected');
      const data = await response.json();

      setHistory((prev) => [...prev, { role: 'bot', text: data.response }]);
    } catch (error) {
      setHistory((prev) => [...prev, { role: 'bot', text: 'Error: Failed to fetch risk intelligence response.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '20px auto', fontFamily: 'sans-serif', border: '1px solid #ccc', borderRadius: '8px', padding: '16px' }}>
      <h3>LendShield AI — Credit Context Copilot</h3>
      <div style={{ marginBottom: '12px' }}>
        <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Application ID: </label>
        <input type="text" value={applicationId} onChange={(e) => setApplicationId(e.target.value)} style={{ padding: '4px', borderRadius: '4px', border: '1px solid #aaa' }} />
      </div>
      <div style={{ height: '300px', overflowY: 'scroll', border: '1px solid #eee', padding: '8px', marginBottom: '12px', background: '#f9f9f9', borderRadius: '4px' }}>
        {history.map((msg, idx) => (
          <div key={idx} style={{ textAlign: msg.role === 'user' ? 'right' : 'left', margin: '8px 0' }}>
            <span style={{ background: msg.role === 'user' ? '#0070f3' : '#e1e1e1', color: msg.role === 'user' ? '#fff' : '#000', padding: '6px 12px', borderRadius: '12px', display: 'inline-block' }}>
              {msg.text}
            </span>
          </div>
        ))}
        {loading && <div style={{ color: '#888', fontSize: '12px' }}>Analyzing credit context parameters...</div>}
      </div>
      <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '8px' }}>
        <input type="text" placeholder="Ask about default probabilities or risk indicators..." value={message} onChange={(e) => setMessage(e.target.value)} style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }} />
        <button type="submit" disabled={loading} style={{ padding: '8px 16px', background: '#0070f3', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Send</button>
      </form>
    </div>
  );
}
 
