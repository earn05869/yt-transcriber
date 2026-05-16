/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.html", "./*.js"],
  theme: {
    extend: {
      colors: {
        'brand-bg': '#121212',
        'brand-card': '#1E1E1E',
        'brand-border': '#333333',
      }
    },
  },
  plugins: [],
}
