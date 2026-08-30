/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emits .next/standalone with a self-contained server.js and only the
  // node_modules actually reached at runtime, so the runtime image does not
  // need to carry the full dependency tree.
  output:  process.env.VERCEL ? undefined : "standalone",
  // Next 16 writes its own AGENTS.md / CLAUDE.md into the project on dev start.
  // This project already has HANDOVER.md as its single source of truth, so the
  // generated boilerplate is noise.
  agentRules: false,
};

export default nextConfig;
