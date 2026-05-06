import SiteHeader from "@/components/SiteHeader";
import AskPanel from "@/components/AskPanel";

export default function AskPage() {
  return (
    <div className="app-shell">
      <SiteHeader />
      <main className="stack">
        <AskPanel />
      </main>
    </div>
  );
}
