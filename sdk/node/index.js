const SDK_VERSION = "0.1.0-alpha.1";

if (!SDK_VERSION) {
  process.exit(1);
}

console.log(JSON.stringify({ sdkVersion: SDK_VERSION }));
