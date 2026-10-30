let inMemoryCases = [];

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method === 'POST') {
    const newCase = req.body;
    if (newCase && newCase.case_id) {
      inMemoryCases = [newCase, ...inMemoryCases.filter((c) => c.case_id !== newCase.case_id)].slice(0, 50);
    }
    return res.status(201).json({ success: true });
  }

  return res.status(200).json(inMemoryCases);
}
