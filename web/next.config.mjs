/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Next 16 writes its own AGENTS.md / CLAUDE.md into the project on dev start.
  // This project already has HANDOVER.md as its single source of truth, so the
  // generated boilerplate is noise.
  agentRules: false,
};

export default nextConfig;
