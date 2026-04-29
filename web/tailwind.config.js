/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        edge: '#1e3a8a',
        cloud: '#7c3aed',
        ok: '#16a34a',
        block: '#dc2626'
      }
    }
  },
  plugins: []
};
