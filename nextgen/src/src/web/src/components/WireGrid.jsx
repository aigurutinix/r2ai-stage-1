// Nền lưới phối cảnh (wireframe) mờ phía sau, giống mockup. Thuần SVG/CSS, không ảnh ngoài.
export default function WireGrid() {
  return (
    <div className="wiregrid" aria-hidden="true">
      <svg width="100%" height="100%" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1b3a8a" stopOpacity="0.0" />
            <stop offset="55%" stopColor="#1b3a8a" stopOpacity="0.10" />
            <stop offset="100%" stopColor="#2b56c8" stopOpacity="0.28" />
          </linearGradient>
          <pattern id="grid" width="46" height="46" patternUnits="userSpaceOnUse">
            <path d="M46 0 L0 0 0 46" fill="none" stroke="#3b6fd4" strokeOpacity="0.18" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        <rect width="100%" height="100%" fill="url(#fade)" />
      </svg>
    </div>
  );
}
