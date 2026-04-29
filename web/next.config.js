/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
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
