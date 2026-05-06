import SiteHeader from "@/components/SiteHeader";
import UploadPanel from "@/components/UploadPanel";

export default function UploadPage() {
  return (
    <div className="app-shell">
      <SiteHeader />
      <main className="stack">
        <UploadPanel />
      </main>
    </div>
  );
}
