import './globals.css';
import Link from 'next/link';

export const metadata = {
  title: 'Hybrid Multi-Agent Demo',
  description: 'Edge SLM + Cloud LLM with a privacy boundary'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="border-b border-slate-800 bg-slate-950 px-6 py-3">
          <div className="mx-auto flex max-w-5xl items-center gap-6">
            <span className="text-sm font-bold text-slate-200">Hybrid Multi-Agent</span>
            <Link href="/" className="text-sm text-slate-400 hover:text-slate-100">
              Demo
            </Link>
            <Link href="/status" className="text-sm text-slate-400 hover:text-slate-100">
              Service Status
            </Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
