/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Disable Next.js' built-in gzip compression. Compression buffers small
  // chunks (like SSE events) until a threshold, so per-stage events would
  // only reach the browser when the upstream stream ends.
  compress: false,
  experimental: {
    typedRoutes: false
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://localhost:8000';
    return [
      { source: '/api/:path*', destination: `${backend}/api/:path*` },
      { source: '/agui', destination: `${backend}/agui` }
    ];
  }
};
module.exports = nextConfig;
