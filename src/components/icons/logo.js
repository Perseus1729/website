import React from 'react';

const IconLogo = () => (
  <svg id="logo" xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 84 96">
    <title>Logo</title>
    <g transform="translate(-8, -2)">
      <g transform="translate(11, 5)">
        {/* Hexagon */}
        <polygon
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points="39 0 0 22 0 67 39 90 78 68 78 23"
        />
        {/* Larger Letter H Centered */}
        <g transform="translate(22, 33) scale(1.3)"> 
          <path
            d="M5 0 V24 H10 V12 H20 V24 H25 V0 H20 V10 H10 V0 Z"
            fill="currentColor"
          />
        </g>
      </g>
    </g>
  </svg>
);

export default IconLogo;
